import torch
import argparse

from distill.provider import NeRFDataset
from distill.gui import StyleRFVolVisGUI
from distill.utils import StyleRFVolVisTrainer
from distill.utils import *
from distill.network import StyleRFVolVis
from functools import partial
from loss import huber_loss
from icecream import install
import re
install()
# ic.disable()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('nerf_path', type=str)
    parser.add_argument('--withoutLighting', action='store_true', default=False,help="not use lighting when inference non-photo stylized result")
    parser.add_argument('--evalTime', action='store_true', default=False,help="eval render time or not")
    parser.add_argument('--config', help="configuration file", type=str, required=False)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--video', action='store_true', help="video mode")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--version_id', type=int, default=-1, help="palettenerf's version id")
    
    ###* genral training options
    parser.add_argument('--val_interval', type=int, default=1, help="iter") #* nunmber of iterations for non-photorealistic stylization

    ###* training options for non- and photorealistic options
    parser.add_argument('--recolor', action='store_false', help="Recolor the palette for non-photorealistic stylization") #* enbale recolor default: False
    parser.add_argument('--iters', type=int, default=5000, help="training iters")  #was 30000
    parser.add_argument('--stage1_iters', type=int, default=2000, help="stage 1 iters") #* new added
    parser.add_argument('--style_epoch', type=int, default=5, help="training iters") #* nunmber of iterations for non-photorealistic stylization
    
    parser.add_argument('--use_real_data_for_train', action='store_true', help="generate novel camera pose for distillation")
    # parser.add_argument('-s','--styleDirs', action='append', help='selected style dirs', required=True)
    parser.add_argument('-s','--styleDirs', nargs='+', help='selected style dirs', required=True)
    parser.add_argument("--loss_rate_fea_sc", type=float, default=0.002)
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--minlr', type=float, default=1e-3, help="initial learning rate")
    
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--batch_size', type=int, default=4096, help="non-photorealistic stylization rays batch size") #* added
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")
    parser.add_argument('--random_size', type=int, default=0, help="[experimental] rendom size for image space smoothing")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', choices=['srgb', 'linear'], help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")
    parser.add_argument('--datatype', type=str, choices=['llff', 'blender', 'mip360'], help="Type of dataset (used for testing views generation)")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=960, help="GUI width")
    parser.add_argument('--H', type=int, default=540, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    parser.add_argument('--continue_training', action='store_true', help="continue training")

    ### Palette Extraction options
    parser.add_argument('--extract_palette', action='store_true', help="extract palette")
    # parser.add_argument('--use_normalized_palette', action='store_true', help="use palette with normalized input")
    parser.add_argument('--error_thres', type=float, default=5.0/255, help='error threshold for palette extraction')
    parser.add_argument('--update_grid', action='store_true', help="update density grid")
    parser.add_argument("--num_basis", type=int, default=4, help='number of basis') #useless

    # PaletteNeRF options   
    parser.add_argument("--use_initialization_from_rgbxy", action='store_true', help='if specified, use initialization from rgbxy')
    parser.add_argument("--max_freeze_palette_epoch", type=int, default=100, help='epoch number when palette color is released')
    parser.add_argument("--smooth_loss_start_epoch", type=int, default=30, help='epoch number when smooth loss is added')

    parser.add_argument("--lambda_sparsity", type=float, default=2e-4, help='weight of sparsity loss')
    parser.add_argument("--lambda_smooth", type=float, default=4e-3, help='weight of smooth loss')
    parser.add_argument("--lambda_patchsmooth", type=float, default=0, help='weight of smooth loss')
    parser.add_argument("--lambda_view_dep", type=float, default=0.1, help='weight of view dependent color regularity loss')
    parser.add_argument("--lambda_offsets", type=float, default=0.03, help='weight of color offsets regularity loss')
    parser.add_argument("--lambda_weight", type=float, default=0.05, help='weight of weight loss')
    parser.add_argument("--lambda_palette", type=float, default=0.001, help='weight of weight loss')
    
    parser.add_argument("--smooth_sigma_xyz", type=float, default=0.005, help='sigma of 3D position (used in smooth loss)')
    parser.add_argument("--smooth_sigma_color", type=float, default=0.2, help='sigma of color (used in smooth loss)')
    parser.add_argument("--smooth_sigma_clip", type=float, default=0, help='sigma of clip feature (used in smooth loss)')

    parser.add_argument("--lweight_decay_epoch", type=int, default=100, help='epoch number when lambda weight drops to 0')
    
    # CLIP feat options
    parser.add_argument('--pred_clip', action='store_true', help="predict clip featuer")
    parser.add_argument("--clip_dim", type=int, default=16, help='dimension of clip feature')

    opt = parser.parse_args()



    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True
    
    if opt.patch_size > 1:
        opt.error_map = False # do not use error_map if use patch-based training
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."

    if "version" not in os.path.basename(opt.nerf_path):
        nerf_list = glob.glob("%s/version*"%opt.nerf_path)
        # find the newest nerf version
        nerf_id = max([0] + [int(x.split("_")[-1]) for x in nerf_list])
        opt.nerf_path = "%s/version_%d"%(opt.nerf_path, nerf_id) 
    
    #* work space setup
    distill_workspace=opt.nerf_path.replace("results_palette", "results_distill")
    palette_paths= glob.glob("%s/normalized_version*"%os.path.dirname(distill_workspace))
    if (palette_paths == []) or ("normalized" not in palette_paths[0]):
        palette_dir_path = os.path.dirname(opt.nerf_path)
        dir_list = glob.glob("%s/normalized_version*"%palette_dir_path)
        palette_id = max([0] + [int(x.split("_")[-1]) for x in dir_list])
        palette_path = "%s/normalized_version_%d"%(palette_dir_path, palette_id)
        dest_path = os.path.dirname(distill_workspace) + "/normalized_version_%d"%(palette_id)
        import shutil
        shutil.copytree(palette_path, dest_path)
    # read palette
    palette_dir_path = glob.glob("%s/normalized_version*"%os.path.dirname(distill_workspace))[-1]
    
    assert os.path.exists(os.path.join(palette_dir_path, 'extract-palette.txt')), "Extracted palette is missing."
    original_palette = np.loadtxt(os.path.join(palette_dir_path, 'extract-palette.txt'))
    
    # palette_arr_path = os.path.join(palette_dir_path, "palette.npz")
    # original_palette = np.load(palette_arr_path)['palette']
    opt.num_basis = original_palette.shape[0]
    # if len(original_palette.shape) == 1: opt.num_basis = 1


    
    # exit()
    os.makedirs(distill_workspace, exist_ok=True)
   
    workspace_dir = os.path.dirname(distill_workspace)

    if opt.version_id >= 0:
        workspace = "%s/version_%d"%(workspace_dir, opt.version_id)
    else:
        workspace_list = glob.glob("%s/version*"%workspace_dir)
        # find the newest palettenerf version
        workspace_id = max([0] + [int(x.split("_")[-1]) for x in workspace_list])
        workspace = "%s/version_%d"%(workspace_dir, (1-max(opt.test, opt.continue_training))+workspace_id) 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed_everything(opt.seed)

    # load StyleRFVolVis
    model = StyleRFVolVis(
            opt=opt,
            encoding="hashgrid",
            bound=opt.bound,
            cuda_ray=opt.cuda_ray,
            density_scale=1,
            min_near=opt.min_near,
            density_thresh=opt.density_thresh,
            bg_radius=opt.bg_radius,
            original_palette=original_palette, #! delete or set this to None for final version of Code, this was used for debugging
        )
    
    # setting the trainer
    criterion = torch.nn.MSELoss(reduction='none')
    
    if opt.test:
        
        metrics = [PSNRMeter(), SSIMMeter(device=device), LPIPSMeter(device=device)]
        trainer = StyleRFVolVisTrainer('palette', opt, model, device=device, workspace=workspace, fp16=opt.fp16,
                                criterion=criterion, metrics=metrics, use_checkpoint=opt.ckpt)
        
        if opt.gui:    
            test_loader = NeRFDataset(opt, device=device, type='traintest').dataloader()
            try:
                video_loader = NeRFDataset(opt, device=device, type='video').dataloader()
            except:
                print("Loading video poses failed. Skipped.")
                video_loader = None
            opt.H = test_loader._data.H
            opt.W = test_loader._data.W
            # train
            train_loader = NeRFDataset(opt, device=device, type='train').dataloader()
            valid_loader = NeRFDataset(opt, device=device, type='val', downscale=1).dataloader()
            gui = StyleRFVolVisGUI(opt, trainer, train_loader=train_loader,valid_loader=valid_loader, video_loader=video_loader)
            gui.render()
        else:
            trainer.model.photo_mode = False #* use student color network for non-photo stylization rendering
            test_loader = NeRFDataset(opt, device=device, type='test').dataloader()
            if test_loader.has_gt:
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
            trainer.test(test_loader, write_video=True, eval_time=True) # test and save video
    else:
        
        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
        scheduler = None
        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = StyleRFVolVisTrainer('palette', opt, model, device=device, workspace=workspace, optimizer=optimizer, criterion=criterion, 
            ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=metrics, 
            use_checkpoint=opt.ckpt, palette_path=opt.nerf_path, eval_interval=opt.val_interval)
        
        # train
        train_loader = NeRFDataset(opt, device=device, type='train').dataloader()
        valid_loader = NeRFDataset(opt, device=device, type='val', downscale=1).dataloader()
        max_epoch_Photo = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        max_epoch_nonPhoto = opt.style_epoch
        
        # test
        test_loader = NeRFDataset(opt, device=device, type='test', n_test=30).dataloader()
        
        if opt.gui:    
            # test_loader = NeRFDataset(opt, device=device, type='traintest').dataloader()
            try:
                video_loader = NeRFDataset(opt, device=device, type='video').dataloader()
            except: 
                print("Loading video poses failed. Skipped.")
                video_loader = None
            opt.H = train_loader._data.H
            opt.W = train_loader._data.W
            gui = StyleRFVolVisGUI(opt, trainer, train_loader=train_loader,valid_loader=valid_loader, video_loader=video_loader)
            gui.render()
        else:
            trainer.distill(train_loader, valid_loader, max_epoch_Photo)
            trainer.stylize(valid_loader, max_epoch_nonPhoto)
            trainer.test(test_loader, write_video=True) # test and save vide