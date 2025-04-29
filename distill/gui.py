from configparser import Interpolation
import math
import torch
import torch.optim as optim
import numpy as np
import pickle
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R
from dearpygui_ext.themes import create_theme_imgui_light
import PIL
from nerf.utils import *
from .renderer import RegionEdit, Stylizer, PhotorealisticStylizer
# from fdialog import FileDialog

def getRefernceStyleImg(styleW,styleH,imgPath,maskPath):
    mask_color = np.array([30, 144, 255, 0.6])
    image = PIL.Image.open(imgPath).convert("RGBA")
    image_arr = np.array(image)
    ori_imgW, ori_imgH = image_arr.shape[1], image_arr.shape[0]
    mask = np.load(maskPath)
    mask = np.zeros((ori_imgH, ori_imgW), dtype=np.float32)
    
    composite_image = PIL.Image.new("RGBA", image.size, color=(int(mask_color[0]), int(mask_color[1]), int(mask_color[2]), int(mask_color[3]* 255)))
    composite_image = np.array(composite_image)*mask[:, :, np.newaxis]
    composite_image = PIL.Image.fromarray(composite_image.astype(np.uint8))
    composite_image = PIL.Image.alpha_composite(image, composite_image)
    

    output_img = PIL.Image.new("RGB", (styleW,styleH), color=(255, 255, 255))
    
    if ori_imgW > ori_imgH:
        imgW = int(styleW)
        imgH = int(ori_imgH / ori_imgW * styleW)
        composite_image = composite_image.resize((imgW, imgH))
        # output_img.paste(composite_image, (0, (styleH - imgH) // 2))
        output_img.paste(composite_image, (0, 50))
    else:
        imgH = int(styleH)
        imgW = int(ori_imgW / ori_imgH * styleH)
        composite_image = composite_image.resize((imgW, imgH))
        output_img.paste(composite_image, ((styleW - imgW) // 2, 0))
    return np.array(output_img)

class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r # camera distance from center
        self.fovy = fovy # in degree
        self.center = np.array([0, 0, 0], dtype=np.float32) # look at this point
        self.rot = R.from_quat([1, 0, 0, 0]) # init camera matrix: [[1, 0, 0], [0, -1, 0], [0, 0, 1]] (to suit ngp convention)
        self.up = np.array([0, 1, 0], dtype=np.float32) # need to be normalized!

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res
    
    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])
    
    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0] # why this is side --> ? # already normalized.
        rotvec_x = self.up * np.radians(-0.1 * dx)
        rotvec_y = side * np.radians(-0.1 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, dy, dz])

class DataValues():
    def __init__(self):
        self.selected_palette_id = 0

class StyleRFVolVisGUI:
    def __init__(self, opt, trainer, train_loader=None,valid_loader=None, video_loader=None, debug=True, distilled=False, stylized=False,max_epoch_Photo=10, max_epoch_nonPhoto=5):
        self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        self.debug = debug
        self.bg_color = torch.ones(3, dtype=torch.float32) # default white bg
        self.training = False
        self.step = 0 # training step 
        self.max_epoch_Photo = max_epoch_Photo
        self.max_epoch_nonPhoto = max_epoch_nonPhoto
        

        self.trainer = trainer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.video_loader = video_loader
        if train_loader is not None:
            self.trainer.error_map = train_loader._data.error_map

        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.selected_point = None
        self.selected_pixel = None
        self.xyz = None
        self.clip_feat = None
        self.need_update = True # camera moved, should reset accumulation
        self.spp = 1 # sample per pixel
        self.mode = 'image' # choose from ['image', 'depth']
        self.stu_color=[False for i in range(opt.num_basis)] if not opt.test else [True for i in range(opt.num_basis)]
        
        self.styleImgW = 800//2
        self.styleImgH = 780//2

        self.dynamic_resolution = False
        self.downscale = 1
        self.train_steps = 16
        
        self.cur_epoch = 0
        self.stylize = False
        self.need_optimize_stylize = False
        self.cached_stylizer = None
        self.lambda_ARAP = 0.1
        self.style_point_list = []
        self.style_color_list = []
        self.style_pixel = None
        self.style_image = [np.ones((self.styleImgW, self.styleImgH, 3), dtype=np.float32) for i in range(opt.num_basis)]
        self.drawed_style_image = np.ones((self.styleImgW, self.styleImgH, 3), dtype=np.float32)
        self.style_W = 0
        self.style_H = 0
        
        self.distilled = distilled
        self.stylized = stylized
        
        # self.trainer.model.edit = RegionEdit(opt)
        self.trainer.model.edit = PhotorealisticStylizer(opt) #* just initialize, maybe not use
        self.style_dirs = self.trainer.model.edit.style_dirs #* also just initialize, maybe not use
        # print(self.style_dirs)
        self.load_palette()
        if self.opt.test:
            self._init_style_texture()

        dpg.create_context()
        self.register_dpg()
        self.test_step()
    
    def _init_style_texture(self):
        self.drawed_style_image = np.ones((self.styleImgW, self.styleImgH, 3), dtype=np.float32)
        for i in range(self.opt.num_basis):
            #load style image
            ImgPath = "/home/dullpigeon/Desktop/StyleProj/StyleRF-VolVis/styles/17/style17.png"
            # ImgPath = os.path.join(self.style_dirs[i], 'img.png')
            maskPath = os.path.join(self.style_dirs[i], 'mask.npy')
            image = getRefernceStyleImg(self.styleImgW,self.styleImgH,ImgPath,maskPath)
            self.style_image[i] = (image/255.0).clip(0, 1).astype(np.float32)
        
    def load_palette(self):
        self.weight_mode = False
        self.palette = self.trainer.model.basis_color.clone()
        self.origin_palette = self.palette.clone()
        self.highlight_palette_id = 0
        
    def __del__(self):
        dpg.destroy_context()


    def train_step(self):
        self.trainer.model.edit.style_dirs = self.style_dirs
        self.trainer.model.edit.parse_style()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        outputs = self.trainer.train_gui(self.cur_epoch,self.trainer.model.edit,self.train_loader,self.valid_loader,self.max_epoch_Photo,self.max_epoch_nonPhoto,self.cam.pose, self.cam.intrinsics, self.W, self.H, self.bg_color, self.spp, self.downscale, gui_mode=True,stu_color=self.stu_color)
        self.cur_epoch += 1
        if self.cur_epoch > self.max_epoch_Photo + self.max_epoch_nonPhoto:
            self.training = False
            self.cur_epoch = 0
        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        # update dynamic resolution
        if self.dynamic_resolution:
            # max allowed infer time per-frame is 200 ms
            full_t = t / (self.downscale ** 2)
            downscale = min(1, max(1/4, math.sqrt(100 / full_t)))
            if downscale > self.downscale * 1.2 or downscale < self.downscale * 0.8:
                self.downscale = downscale

        output_buffer = self.prepare_buffer(outputs)

        self.render_buffer = output_buffer
        self.spp = 1
        self.need_update = False

        dpg.set_value("_log_infer_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
        dpg.set_value("_log_resolution", f'{int(self.downscale * self.W)}x{int(self.downscale * self.H)}')
        # dpg.set_value("_log_spp", self.spp)
        dpg.set_value("_texture", self.render_buffer)
        # dpg.set_value("_style_texture", self.style_image[self.highlight_palette_id])
            
        # dpg.set_value("_style_texture", self.style_image[self.highlight_palette_id]) #todo: uncomment this line

    def prepare_buffer(self, outputs):
        if self.mode == 'image':
            return outputs['image']
        else:
            return np.expand_dims(outputs['depth'], -1).repeat(3, -1)

    def test_step(self):
        # TODO: seems we have to move data from GPU --> CPU --> GPU?    
        max_spp = self.opt.max_spp if self.dynamic_resolution else 1
        if self.need_update or self.spp < max_spp:
        
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            outputs = self.trainer.test_gui(self.cam.pose, self.cam.intrinsics, self.W, self.H, self.bg_color, self.spp, self.downscale, gui_mode=True,stu_color=self.stu_color)

            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            # update dynamic resolution
            if self.dynamic_resolution:
                # max allowed infer time per-frame is 200 ms
                full_t = t / (self.downscale ** 2)
                downscale = min(1, max(1/4, math.sqrt(100 / full_t)))
                if downscale > self.downscale * 1.2 or downscale < self.downscale * 0.8:
                    self.downscale = downscale

            output_buffer = self.prepare_buffer(outputs)
            if self.selected_pixel is not None:
                y, x = self.selected_pixel
                self.xyz = torch.from_numpy(outputs['xyz'][x, y]).type_as(self.palette)
                self.clip_feat = torch.from_numpy(outputs['clip_feat'][x, y]).type_as(self.palette)
                self.selected_point = self.xyz
                self.selected_pixel = None

            if self.need_update:
                self.render_buffer = output_buffer
                self.spp = 1
                self.need_update = False
            else:
                self.render_buffer = (self.render_buffer * self.spp + output_buffer) / (self.spp + 1)
                self.spp += 1
            
            dpg.set_value("_log_infer_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
            dpg.set_value("_log_resolution", f'{int(self.downscale * self.W)}x{int(self.downscale * self.H)}')
            # dpg.set_value("_log_spp", self.spp)
            dpg.set_value("_texture", self.render_buffer)
            # dpg.set_value("_style_texture", self.style_image[self.highlight_palette_id])
            
        # dpg.set_value("_style_texture", self.style_image[self.highlight_palette_id]) #todo: uncomment this line
        dpg.set_value("_style_texture", self.drawed_style_image)            #todo: commnet this
        # selected_point = self.selected_point
        # if selected_point is not None:
        #     selected_point = selected_point.detach().cpu().numpy()
        # dpg.set_value("_img_point", "Image Point: " + str(selected_point))
        # dpg.set_value("_style_pixel", "Style Pixel: " + str(self.style_pixel))
            
    def register_dpg(self):

        ### register texture 

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")
            
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.styleImgW, self.styleImgH, self.drawed_style_image, format=dpg.mvFormat_Float_rgb, tag="_style_texture")

        ### register window

        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):

            # add the texture
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        #*-------------------------------------control window-------------------------------------*#
        # control window Control Panel"
        with dpg.window(label="Control Panel", tag="_control_window", width=400, height=self.H//2-10, pos=(self.W, self.H//2+10)):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (161, 238, 189)) #(139, 205, 162)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (174, 255, 204))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (205, 250, 219)) #(174, 255, 203)
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # time
            # if not self.opt.test:
            #     with dpg.group(horizontal=True):
            #         dpg.add_text("Train time: ")
            #         dpg.add_text("no data", tag="_log_train_time")                    

            with dpg.group(horizontal=True):
                dpg.add_text("inference time: ")
                dpg.add_text("no data", tag="_log_infer_time")
           
            with dpg.collapsing_header(label="Photorealistic Stylization", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_text("palette: ")
                    def refresh_palette_color():
                        highlight_color = (self.palette[self.highlight_palette_id].detach().cpu().numpy()*255).clip(0, 255).astype(np.uint8)
                        self.trainer.model.edit.update_delta_hsv(self.origin_palette, self.palette)
                        #self.trainer.model.basis_color.data = self.palette.type_as(self.trainer.model.basis_color.data)
                        dpg.set_value("_palette_color_editor", tuple(highlight_color))

                    def callback_reset_palette(sender, app_data):
                        self.palette = self.origin_palette.clone()
                        self.trainer.model.basis_color = torch.nn.Parameter(self.origin_palette.clone(),requires_grad=False)
                        self.trainer.model.view_dep_weight = [1 for _ in range(len(self.palette))]
                        self.trainer.model.offset_weight = [1 for _ in range(len(self.palette))]
                        self.trainer.model.density_weight = [1 for _ in range(len(self.palette))]
                        # dpg.set_value("_offsets_weight", 1)
                        dpg.set_value("_view_dep_weight", 1)
                        dpg.set_value("_density_weight", 1)
                        refresh_palette_color()
                        self.need_update = True
                        
                    dpg.add_button(label="reset", tag="_button_reset_palette", callback=callback_reset_palette)
                    dpg.bind_item_theme("_button_reset_palette", theme_button)
                    
                    def callback_save_palette(sender, app_data):
                        pred_basis_color = []

                        for i in range(self.opt.num_basis):
                            basis_color = self.palette[i,None,None,:].repeat(100, 100, 1)
                            basis_color = basis_color.clamp(0, 1)
                            pred_basis_color.append(basis_color.detach().cpu().numpy())

                        pred_basis_color = (np.concatenate(pred_basis_color, axis=1).clip(0,1)*255).astype(np.uint8)

                        cv2.imwrite(os.path.join("./results_gui", f'basis_color.png'), pred_basis_color[...,[2,1,0]])
                        
                    dpg.add_button(label="save_palette", tag="_button_save_palette", callback=callback_save_palette)
                    dpg.bind_item_theme("_button_save_palette", theme_button)
                
                def callback_set_palette_id(sender, app_data):
                    self.highlight_palette_id = app_data                    
                    refresh_palette_color()
                    dpg.set_value("_view_dep_weight", self.trainer.model.view_dep_weight[self.highlight_palette_id])
                    dpg.set_value("_density_weight", self.trainer.model.density_weight[self.highlight_palette_id])
                    dpg.set_value("_select_style_text", f"style for palette {self.highlight_palette_id}:")
                    # dpg.set_value("_style_texture", self.style_image[self.highlight_palette_id]) #todo: uncomment this line
                    dpg.set_value("_button_stu_color", self.stu_color[self.highlight_palette_id])
                    self.need_update = True

                dpg.add_slider_int(label="palette_ID", min_value=0, max_value=len(self.palette)-1, format="%d", 
                                    default_value=self.highlight_palette_id, callback=callback_set_palette_id)
                
                def call_back_set_density_weight(sender, app_data):
                    self.trainer.model.density_weight[self.highlight_palette_id] = app_data
                    self.need_update = True
                dpg.add_slider_float(label="density_weight", tag="_density_weight", min_value=0, max_value=10, format="%f", 
                                    default_value=1, callback=call_back_set_density_weight)
                
                
                def call_back_set_view_dep_weight(sender, app_data):
                    self.trainer.model.view_dep_weight[self.highlight_palette_id] = app_data
                    self.need_update = True
                dpg.add_slider_float(label="specular_weight", tag="_view_dep_weight", min_value=0, max_value=2.5, format="%f", 
                                    default_value=1, callback=call_back_set_view_dep_weight)
                
                def callback_change_palette(sender, app_data):
                    self.palette[self.highlight_palette_id] = torch.tensor(app_data[:3], dtype=torch.float32) # only need RGB in [0, 1]
                    self.trainer.model.basis_color = self.trainer.model.stylizer.recolor(self.palette)
                    refresh_palette_color()
                    self.need_update = True

                highlight_color = (self.palette[self.highlight_palette_id].detach().cpu().numpy()*255).clip(0, 255).astype(np.uint8)
                dpg.add_color_edit(tuple(highlight_color), label="palette color", width=200, tag="_palette_color_editor", 
                                    no_alpha=True, callback=callback_change_palette)
            #*Non-Photorealistic Stylization
            # if not self.opt.test:
            with dpg.collapsing_header(label="Non-Photorealistic Stylization", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_text(f"style for palette {self.highlight_palette_id}:",tag="_select_style_text")
                    def callback_select_style_image(sender, app_data):
                        dirPath = app_data['selections'][list(app_data['selections'].keys())[0]]
                        dirPath = os.path.dirname(dirPath) #* this is a fix for selection bug of dearpygui
                        self.style_dirs[self.highlight_palette_id] = dirPath
                        #check is this a valid style directory
                        check_items = ['img.png', 'mask.npy', "valid_img.png", 'valid_mask.npy', 'color.npy']
                        for item in check_items:
                            if not os.path.exists(os.path.join(dirPath, item)):
                                print("Invalid style directory")
                                return
                        #load style image
                        ImgPath = os.path.join(dirPath, 'img.png')
                        maskPath = os.path.join(dirPath, 'mask.npy')
                        image = getRefernceStyleImg(self.styleImgW,self.styleImgH,ImgPath,maskPath)
                        
                        self.style_image[self.highlight_palette_id] = (image/255.0).clip(0, 1).astype(np.float32)
                        self.drawed_style_image = self.style_image[self.highlight_palette_id]
                
                    with dpg.file_dialog(directory_selector=True, show=False, width=800, height=300, default_path="/home/dullpigeon/Desktop/StyleProj/StyleRF-VolVis/styles", callback=callback_select_style_image, tag="file_dialog_id"):
                        dpg.add_file_extension("", color=(0, 0, 0, 255))

                    dpg.add_button(label="select style dir", tag="_button_select_style_dir", callback=lambda: dpg.show_item("file_dialog_id"))
                    dpg.bind_item_theme("_button_select_style_dir", theme_button)
                        
                    def callback_delete_correspondence(sender, app_data):
                        idx = int(sender.split("_")[-1])
                        self.style_color_list.pop(idx)
                        self.style_point_list.pop(idx)
                        update_correspondence_list()
                        
                    def update_correspondence_list():
                        dpg.delete_item("_correspondence_list", children_only=True)
                        for i in range(len(self.style_color_list)):
                            dpg.add_text(f"Point: {self.style_point_list[i]}", parent="_correspondence_list")
                            dpg.add_text(f"Color: {self.style_color_list[i]}", parent="_correspondence_list")
                            dpg.add_button(label=f"Delete", parent="_correspondence_list", tag=f"_corr_delete_{i}", callback=callback_delete_correspondence)
                            dpg.bind_item_theme(f"_corr_delete_{i}", theme_button)
                            
                    with dpg.group(horizontal=True):                    
                            
                        def callback_load_correspondence(sender, app_data):
                            assert(self.style_point_list is not None and self.style_color_list is not None)
                            filename = app_data['selections'][list(app_data['selections'].keys())[0]]
                            corr_dict = pickle.load(open(filename, "rb"))
                            self.style_point_list = corr_dict['points']
                            self.style_color_list = corr_dict['colors']
                            update_correspondence_list()
                            
                        with dpg.file_dialog(directory_selector=False, show=False, width=800, height=300, 
                                            default_path="./results_gui", callback=callback_load_correspondence, tag="style_file_dialog"):
                            dpg.add_file_extension("", color=(255, 150, 150, 255))
                            dpg.add_file_extension(".*")
                            dpg.add_file_extension(".pkl", color=(255, 0, 255, 255), custom_text="[pkl]")

                # train / stop
                with dpg.group(horizontal=True):
                    dpg.add_text("train: ")

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            self.training = True
                            dpg.configure_item("_button_train", label="start")

                    dpg.add_button(label="start", tag="_button_train", callback=callback_train)
                    dpg.bind_item_theme("_button_train", theme_button)

                    def callback_reset(sender, app_data):
                        @torch.no_grad()
                        def weight_reset(m: nn.Module):
                            reset_parameters = getattr(m, "reset_parameters", None)
                            if callable(reset_parameters):
                                m.reset_parameters()
                        self.trainer.model.stu_NonViewColorNet.apply(fn=weight_reset)
                        self.trainer.model.encoder_palette_stu.apply(fn=weight_reset)
                        self.trainer.model.reset_extra_state() # for cuda_ray density_grid and step_counter
                        self.need_update = True

                    dpg.add_button(label="reset", tag="_button_reset", callback=callback_reset)
                    dpg.bind_item_theme("_button_reset", theme_button)

                # save ckpt
                with dpg.group(horizontal=True):
                    # dpg.add_text("Use Student Color: ")

                    def callback_stu_color(sender, app_data):
                        self.stu_color[self.highlight_palette_id] = not self.stu_color[self.highlight_palette_id]
                        self.need_update = True

                    dpg.add_checkbox(label="unrestricted color", tag="_button_stu_color", callback=callback_stu_color, default_value=self.stu_color[self.highlight_palette_id])
                    dpg.bind_item_theme("_button_stu_color", theme_button)

                
                # save ckpt
                with dpg.group(horizontal=True):
                    dpg.add_text("checkpoint: ")

                    def callback_save(sender, app_data):
                        self.trainer.save_checkpoint(full=True, best=False)
                        # dpg.set_value("_log_ckpt", "saved " + os.path.basename(self.trainer.stats["checkpoints"][-1]))
                        self.trainer.epoch += 1 # use epoch to indicate different calls.

                    dpg.add_button(label="save", tag="_button_save", callback=callback_save)
                    dpg.bind_item_theme("_button_save", theme_button)

                    # dpg.add_text("", tag="_log_ckpt")

                # with dpg.group(horizontal=True):
                #     dpg.add_text("", tag="_log_train_log")
            
            
            # rendering options
            with dpg.collapsing_header(label="Rendering Options", default_open=False):

                # dynamic rendering resolution
                with dpg.group(horizontal=True):

                    def callback_set_dynamic_resolution(sender, app_data):
                        if self.dynamic_resolution:
                            self.dynamic_resolution = False
                            self.downscale = 1
                        else:
                            self.dynamic_resolution = True
                        self.need_update = True

                    dpg.add_checkbox(label="dynamic resolution", default_value=self.dynamic_resolution, callback=callback_set_dynamic_resolution)
                    dpg.add_text(f"{self.W}x{self.H}", tag="_log_resolution")

                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True
                
                dpg.add_combo(('image', 'depth'), label='mode', default_value=self.mode, callback=callback_change_mode)

                # bg_color picker
                def callback_change_bg(sender, app_data):
                    self.bg_color = torch.tensor(app_data[:3], dtype=torch.float32) # only need RGB in [0, 1]
                    self.need_update = True

                dpg.add_color_edit((255, 255, 255), label="background color", width=200, tag="_color_editor", no_alpha=True, callback=callback_change_bg)
                
                with dpg.group(horizontal=True):
                    def callback_renderview(sender, app_data):
                        rendered_img = self.render_buffer
                        rendered_img = (rendered_img*255).astype(np.uint8)[...,[2,1,0]]
                        cv2.imwrite(os.path.join("./results_gui", f'rendered_img.png'), rendered_img)
                        print("Image Saved")

                    dpg.add_button(label="save image", tag="_button_render_view", callback=callback_renderview)
                    dpg.bind_item_theme("_button_render_view", theme_button)

                    if self.video_loader is not None:
                        def callback_rendervideo(sender, app_data):
                            self.trainer.test(self.video_loader, save_path="./results_gui", write_video=True, gui_mode=True) # test and save video

                        dpg.add_button(label="render video", tag="_button_render_video", callback=callback_rendervideo)
                        dpg.bind_item_theme("_button_render_video", theme_button)
                    
                def callback_set_testcam(sender, app_data):
                    self.test_cam_id = app_data-1
                    test_pose = self.train_loader._data.poses[app_data-1].detach().cpu().numpy()
                    self.cam.rot = R.from_matrix(test_pose[:3, :3])
                    self.cam.radius = 2
                    center = test_pose[:3, :3] @ np.array([0, 0, -self.cam.radius])[...,np.newaxis]
                    self.cam.center = center[:,0] - test_pose[:3, 3]
                    #     @property
                    # def pose(self):
                    #     # first move camera to radius
                    #     res = np.eye(4, dtype=np.float32)
                    #     res[2, 3] -= self.radius
                    #     # rotate
                    #     rot = np.eye(4, dtype=np.float32)
                    #     rot[:3, :3] = self.rot.as_matrix()
                    #     res = rot @ res
                    #     # translate
                    #     res[:3, 3] -= self.center
                    #     return res
                    def intrinsics(self):
                        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
                        return np.array([focal, focal, self.W // 2, self.H // 2])
                    fovy = self.train_loader._data.intrinsics[1] 
                    self.cam.fovy = np.degrees(np.arctan(self.train_loader._data.H / fovy / 2) * 2)
                    self.cam.pose
                    self.need_update = True
                dpg.add_slider_int(label="test_pose", min_value=1, max_value=len(self.train_loader._data.poses), format="%d", default_value=0, callback=callback_set_testcam)

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = app_data
                    self.need_update = True

                dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, format="%d deg", default_value=self.cam.fovy, callback=callback_set_fovy)

                # dt_gamma slider
                def callback_set_dt_gamma(sender, app_data):
                    self.opt.dt_gamma = app_data
                    self.need_update = True

                dpg.add_slider_float(label="dt_gamma", min_value=0, max_value=0.1, format="%.5f", default_value=self.opt.dt_gamma, callback=callback_set_dt_gamma)

                # max_steps slider
                def callback_set_max_steps(sender, app_data):
                    self.opt.max_steps = app_data
                    self.need_update = True

                dpg.add_slider_int(label="max steps", min_value=1, max_value=1024, format="%d", default_value=self.opt.max_steps, callback=callback_set_max_steps)

                # aabb slider
                def callback_set_aabb(sender, app_data, user_data):
                    # user_data is the dimension for aabb (xmin, ymin, zmin, xmax, ymax, zmax)
                    self.trainer.model.aabb_infer[user_data] = app_data

                    # also change train aabb ? [better not...]
                    #self.trainer.model.aabb_train[user_data] = app_data

                    self.need_update = True

                dpg.add_separator()
                dpg.add_text("axis-aligned bounding box:")

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="x", width=150, min_value=-self.opt.bound, max_value=0, format="%.2f", default_value=-self.opt.bound, callback=callback_set_aabb, user_data=0)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound, format="%.2f", default_value=self.opt.bound, callback=callback_set_aabb, user_data=3)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="y", width=150, min_value=-self.opt.bound, max_value=0, format="%.2f", default_value=-self.opt.bound, callback=callback_set_aabb, user_data=1)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound, format="%.2f", default_value=self.opt.bound, callback=callback_set_aabb, user_data=4)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="z", width=150, min_value=-self.opt.bound, max_value=0, format="%.2f", default_value=-self.opt.bound, callback=callback_set_aabb, user_data=2)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound, format="%.2f", default_value=self.opt.bound, callback=callback_set_aabb, user_data=5)
        
            # debug info
            # if self.debug:
            #     with dpg.collapsing_header(label="Debug"):
            #         # pose
            #         dpg.add_separator()
            #         dpg.add_text("Camera Pose:")
            #         dpg.add_text(str(self.cam.pose), tag="_log_pose")
        #*-------------------------------------control window-------------------------------------*#

            
        #*-------------------------------------Style window-------------------------------------*#
                    
        with dpg.window(label="Style Image",tag="_style_window", width=400, height=self.H-(self.H//2-10), pos=(self.W, 0)):
            # add the texture
            dpg.add_image("_style_texture")

            # with dpg.collapsing_header(label="Correspondence List", default_open=True, tag="_correspondence_list"):
            #     pass
        #*-------------------------------------Style window-------------------------------------*#

        ### register camera handler

        def callback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

            # if self.debug:
            #     dpg.set_value("_log_pose", str(self.cam.pose))


        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

            # if self.debug:
            #     dpg.set_value("_log_pose", str(self.cam.pose))


        def callback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

            # if self.debug:
            #     dpg.set_value("_log_pose", str(self.cam.pose))

        def callback_select_point(sender, app_data):

            if not dpg.is_item_focused("_primary_window") and not dpg.is_item_focused("_style_window"):
                return
            style_img = dpg.is_item_focused("_style_window")
            x, y = dpg.get_mouse_pos()
            x = int(x)
            y = int(y)
            if not style_img:
                if x > 0 and y > 0 and x < self.W and y < self.H:
                    self.selected_pixel = [x,y]
                    self.selected_point = None
            else:
                if x > 0 and y > 0 and x < 400 and y < 400:
                    self.style_pixel = [x,y]
                    self.drawed_style_image = cv2.circle(self.style_image.copy(), self.style_pixel, 5, (255, 0, 0), -1)
                        
            self.need_update = True
                
            # if self.debug:
            #     dpg.set_value("_log_pose", str(self.cam.pose))
                
        def callback_clear_point(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return
            print("Unselecting point")
            self.selected_point = None
            self.selected_pixel = None
            self.xyz = None
            self.clip_feat = None
            self.style_pixel = None
            self.need_update = True

            # if self.debug:
            #     dpg.set_value("_log_pose", str(self.cam.pose))

        with dpg.handler_registry():
            # dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=callback_select_point)
            # dpg.add_mouse_double_click_handler(button=dpg.mvMouseButton_Right, callback=callback_clear_point)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan)


        dpg.create_viewport(title='StyleRF-VolVis', width=self.W+self.styleImgW, height=self.H, resizable=True)
        

        ### global theme
        light_theme = create_theme_imgui_light()
        dpg.bind_theme(light_theme)
        
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
                
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (186, 212, 243), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, (186, 212, 243), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (186, 212, 243), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Header, (186, 212, 243), category=dpg.mvThemeCat_Core)
        
        dpg.bind_item_theme("_primary_window", theme_no_padding)
        dpg.bind_item_theme("_style_window", theme_no_padding)
        
        
        
        # dpg.bind_item_theme("_control_window", theme_no_padding)
        
        ### Control Pannel theme
        with dpg.theme() as theme_control:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, (186, 212, 243), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (186, 212, 243), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (186, 212, 243), category=dpg.mvThemeCat_Core)
                
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)
        dpg.bind_item_theme("_control_window", theme_control)
        

        dpg.setup_dearpygui()

        #dpg.show_metrics()

        dpg.show_viewport()


    def render(self):
        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                self.train_step()
            self.test_step()
            dpg.render_dearpygui_frame()