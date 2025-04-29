from importlib.metadata import requires
import math
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import raymarching
from nerf.utils import custom_meshgrid
from .utils import normalize, rgb_to_hsv, hsv_to_rgb
from nerf.utils import srgb_to_linear
#* stylize part
from .nn_loss import *

from PIL import Image
from torchvision import transforms

from matplotlib import pyplot as plt
# Utility functions 

def cos_distance(x, y, safemode=False):
    result = 1-(normalize(x) * normalize(y)).sum(dim=-1)
    if safemode:
        result = result * torch.minimum(x.norm(dim=-1), y.norm(dim=-1)).clip(max=0.2)*5
    return result

def isclose(x, val, threshold = 1e-6):
    return torch.abs(x - val) <= threshold

def safe_pow(x, p):
    sqrt_in = torch.relu(torch.where(isclose(x, 0.0), torch.ones_like(x) * 1e-6, x))
    return torch.pow(sqrt_in, p)

def safe_linear_to_srgb(x):
    sqrt_in = torch.relu(torch.where(isclose(x, 0.0), torch.ones_like(x) * 1e-6, x))
    return 1.055 * sqrt_in ** 0.41666 - 0.055

def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def seed_everything(seed=42):
   """This function will remove the initial uncertainty when init the neural network


   Args:
       seed (int): seed used to init the random components
   """
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True

def plot_pointcloud(pc, color=None):
    # pc: [N, 3]
    # color: [N, 3/4]
    print('[visualize points]', pc.shape, pc.dtype, pc.min(0), pc.max(0))
    pc = trimesh.PointCloud(pc, color)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()
        
# Solver of User-guided Photorealistic style transfer 
class Stylizer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        dI = torch.zeros(opt.num_basis, dtype=torch.float32)
        self.dI = torch.nn.Parameter(dI, requires_grad=True)
        dP = torch.zeros(1, opt.num_basis, 3, dtype=torch.float32)
        self.dP = torch.nn.Parameter(dP, requires_grad=True)
        ddelta = torch.eye(3, dtype=torch.float32)[None,:,:].repeat(opt.num_basis, 1, 1) # N_p x 3 x 3
        self.ddelta = torch.nn.Parameter(ddelta, requires_grad=True)
        
    def ARAP_loss(self):
        I = torch.eye(3, dtype=torch.float32, device=self.ddelta.device)[None,:,:]
        return ((torch.bmm(self.ddelta, self.ddelta.transpose(1, 2)) - I)**2).sum()
    
    def forward(self, radiance, omega, palette, offsets, view_dep=None):
        
        prefix = offsets.shape[:-2]
        radiance = radiance.reshape(-1, 1, 1)
        omega = omega.reshape(-1, self.opt.num_basis, 1)
        palette = palette.reshape(-1, self.opt.num_basis, 3)
        offsets = offsets.reshape(-1, self.opt.num_basis, 3)
        
        palette = (palette+self.dP) # N x N-p x 3
        offsets = torch.einsum("npi, pij->npj", offsets, self.ddelta) # N x N_p x 3
    
        basis_rgb = ((F.softplus(radiance).repeat(1, self.opt.num_basis, 1)+self.dI[None,:,None]).clamp(0)*(palette+offsets)).clamp(0, 1) # N x N_p x 3
        basis_rgb = omega*basis_rgb # N, N_p, 3
        rgbs = basis_rgb.sum(dim=-2) # N, 3
        
        if view_dep is not None:
            rgbs += view_dep.detach() # N, 3
        return rgbs.reshape(*prefix, 3)



class PhotorealisticStylizer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.style_dirs = opt.styleDirs
        self.styles = []
        self.new_rgbs = []
        self.delta_hsv = torch.zeros(self.opt.num_basis, 3)
        self.delta_hsv[...,1:3] = 1
        self.toTensor = transforms.ToTensor()
        self.parse_style()
        #* remember to call update_delta_hsv() before training
        
    def parse_style(self):# I would pass this style info to non-photo as well
        self.new_rgbs = []
        self.styles = []
        for styleIdx, style_dir in enumerate(self.style_dirs):
            style= {}
            # mask = np.load(style_dir+'/mask.npy')
            mask = np.load(style_dir+'/valid_mask.npy')
            color = np.load(style_dir+'/color.npy')
            # img = np.array(Image.open(style_dir+'/img.png'))
            img = np.array(Image.open(style_dir+'/valid_img.png'))
            masked_img = img * mask[...,None]
            luminance_bg_color = self.calculate_luminance(img,mask)
            # luminance_bg_color = 0.0 #black bg
            # luminance_bg_color = 1.0 #white bg
            masked_img[mask==0] = luminance_bg_color*np.array([255,255,255])#[255,255,255] #make it white bg
            # plt.figure()
            # plt.imshow(masked_img)
            # plt.show()
            style['bg_color'] = luminance_bg_color
            style['img'] = img
            style['name'] = 'style'+str(styleIdx)
            style['mask'] = mask
            style['color'] = color
            style['path'] = style_dir
            style['masked_img'] = masked_img
            self.new_rgbs.append(color)
            self.styles.append(style)
        self.new_rgbs = torch.tensor(self.new_rgbs).float()
        
    def calculate_luminance(self,img_arr,mask=None):
        #* if you found code here interesting, check this: https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color
        if mask is not None:
            img_arr = img_arr[mask==1]
        img_arr = img_arr/255
        img_arr = img_arr**2.2
        Y_arr = 0.2126*img_arr[:,0] + 0.7152*img_arr[:,1] + 0.0722*img_arr[:,2]
        Y_arr[Y_arr<=(216/24389)] = (24389/27)*Y_arr[Y_arr<=(216/24389)]
        Y_arr[Y_arr>(216/24389)] = Y_arr[Y_arr>(216/24389)]**(1/3)*116 - 16
        return np.mean(Y_arr)/100
            
    def update_delta_hsv(self, rgb_orig, palette=None):
        '''
            Given the original palettes' rgb and modified palettes' rgb, calculating the change in HSV Space.
            More specifically, difference in H channel and scales in S,V channel
        '''
        
        if (rgb_orig.device != self.delta_hsv.device) or (rgb_orig.device != self.new_rgbs.device):
            self.delta_hsv = self.delta_hsv.type_as(rgb_orig)
            self.new_rgbs = self.new_rgbs.type_as(rgb_orig)
        if palette is None:
            rgb_all = torch.cat([rgb_orig, self.new_rgbs], dim=0)
        else:
            rgb_all = torch.cat([rgb_orig, palette], dim=0)
        hsv_all = rgb_to_hsv(rgb_all)
        hsv_orig = hsv_all[:self.opt.num_basis]
        hsv_new = hsv_all[self.opt.num_basis:]

        self.delta_hsv[:, 0] = torch.fmod((hsv_new[:,0]-hsv_orig[:,0]+360), 360)
        self.delta_hsv[:, 1] = (hsv_new[:,1]/hsv_orig[:,1]+1e-9)
        self.delta_hsv[:, 2] = (hsv_new[:,2]/hsv_orig[:,2]+1e-9)
    
    def recolor(self,basis_color):#* not used, but this will produce similiar result as forward and maybe faster
        hsv = rgb_to_hsv(basis_color)
        if (basis_color.device != self.delta_hsv.device):
            self.delta_hsv = self.delta_hsv.type_as(basis_color)
        hsv_new = hsv.clone()
        hsv_new[...,0] = torch.fmod((hsv[...,0]+self.delta_hsv[...,0]+360), 360)
        hsv_new[...,1] = torch.clip((hsv[...,1]*self.delta_hsv[...,1]), 0)
        hsv_new[...,2] = torch.clip((hsv[...,2]*self.delta_hsv[...,2]), 0)

        rgb_new = hsv_to_rgb(hsv_new)
        return torch.nn.Parameter(rgb_new, requires_grad=False)
    
    def forward(self, rgbs): 
        if self.opt.recolor:
            hsv = rgb_to_hsv(rgbs)
            if (rgbs.device != self.delta_hsv.device):
                self.delta_hsv = self.delta_hsv.type_as(rgbs)
            weight = torch.ones_like(rgbs[...,0:1,0])
            hsv_new = hsv.clone()
            hsv_new[...,0] = torch.fmod((hsv[...,0]+self.delta_hsv[...,0]+360), 360)
            hsv_new[...,1] = torch.clip((hsv[...,1]*self.delta_hsv[...,1]), 0)
            hsv_new[...,2] = torch.clip((hsv[...,2]*self.delta_hsv[...,2]), 0)

            rgb_new = hsv_to_rgb(hsv_new)
            return torch.lerp(rgbs, rgb_new, weight[...,None])
        else:
            return rgbs
# Controller for regional appearance editing
class RegionEdit(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.mean_xyz = None
        self.mean_clip = None
        self.std_xyz = 1
        self.std_clip = 1
        self.weight_mode = False
        self.delta_hsv = torch.zeros(self.opt.num_basis, 3)
        self.delta_hsv[...,1:3] = 1

    def update_cent(self, mean_xyz=None, mean_clip=None):     
        self.mean_xyz = None if mean_xyz is None else mean_xyz[None,...]
        self.mean_clip = None if mean_clip is None else mean_clip[None,...]
    
    def update_std(self, std_xyz=None, std_clip=None):
        if std_xyz is not None:
            self.std_xyz = std_xyz
        if std_clip is not None:
            self.std_clip = std_clip

    def update_delta_hsv(self, rgb_orig, rgb_new):
        '''
            Given the original palettes' rgb and modified palettes' rgb, calculating the change in HSV Space.
            More specifically, difference in H channel and scales in S,V channel
        '''
        if rgb_orig.device != self.delta_hsv.device:
            self.delta_hsv = self.delta_hsv.type_as(rgb_orig)
        rgb_all = torch.cat([rgb_orig, rgb_new], dim=0)
        hsv_all = rgb_to_hsv(rgb_all)
        hsv_orig = hsv_all[:self.opt.num_basis]
        hsv_new = hsv_all[self.opt.num_basis:]

        self.delta_hsv[:, 0] = torch.fmod((hsv_new[:,0]-hsv_orig[:,0]+360), 360)
        self.delta_hsv[:, 1] = (hsv_new[:,1]/hsv_orig[:,1]+1e-9)
        self.delta_hsv[:, 2] = (hsv_new[:,2]/hsv_orig[:,2]+1e-9)

    def forward(self, rgbs, xyz=None, clip_feat=None):
        hsv = rgb_to_hsv(rgbs)
        if rgbs.device != self.delta_hsv.device:
            self.delta_hsv = self.delta_hsv.type_as(rgbs)
        weight = torch.ones_like(rgbs[...,0:1,0])
        
        # Euclidean distance based filtering
        if xyz is not None and self.mean_xyz is not None:
            weight *= torch.exp(-((xyz-self.mean_xyz)**2.).sum(dim=-1, keepdim=True)/self.std_xyz)
            
        # Semantic map based filtering
        if clip_feat is not None and self.mean_clip is not None:
            #temp = ((clip_feat-self.mean_clip)**2).sum(dim=-1, keepdim=True)
            weight *= torch.exp(-((clip_feat-self.mean_clip)**2.).sum(dim=-1, keepdim=True)/self.std_clip)
            # temp /= (self.mean_clip**2+1e-6).sum(dim=-1, keepdim=True)
            #weight *= (temp < self.std_clip).float() # vtorch.exp(-(temp/self.std_clip))

        hsv_new = hsv.clone()
        hsv_new[...,0] = torch.fmod((hsv[...,0]+self.delta_hsv[...,0]+360), 360)
        hsv_new[...,1] = torch.clip((hsv[...,1]*self.delta_hsv[...,1]), 0)
        hsv_new[...,2] = torch.clip((hsv[...,2]*self.delta_hsv[...,2]), 0)

        rgb_new = hsv_to_rgb(hsv_new)
        if self.weight_mode:
            return weight[...,None].repeat(1, self.opt.num_basis, 3) 
        else:
            return torch.lerp(rgbs, rgb_new, weight[...,None])
        
    
class StyleRFVolVisRenderer(nn.Module):
    def __init__(self,
                 opt,
                 bound=1,
                 cuda_ray=False,
                 density_scale=1, # scale up deltas (or sigmas), to make the density grid more sharp. larger value than 1 usually improves performance.
                 min_near=0.2,
                 density_thresh=0.01,
                 bg_radius=-1,
                 ):
        super().__init__()

        self.bound = bound
        self.cascade = 1 + math.ceil(math.log2(bound))
        self.grid_size = 128
        self.density_scale = density_scale
        self.min_near = min_near
        self.density_thresh = density_thresh
        self.bg_radius = bg_radius # radius of the background sphere.
        self.num_basis = opt.num_basis
        self.freeze_basis_color = opt.use_initialization_from_rgbxy
        self.require_smooth_loss = False
        self.color_weight = 0
        self.opt = opt
        self.edit = None
        self.stylizer = PhotorealisticStylizer(opt)
        
        self.num_basis = opt.num_basis #* number of basis color (or segment class number)
        
        #* distill attributes
        self.photo_mode = True
        self.tea_feat = None
        self.stu_feat = None
        self.tea_color = None
        self.stu_color = None
        
        self.view_dep_weight = [1.0 for _ in range(self.opt.num_basis)]
        self.offsets_weight = [1.0 for _ in range(self.opt.num_basis)]
        self.density_weight = [1.0 for _ in range(self.opt.num_basis)]
        
        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        self.basis_color = torch.zeros([self.num_basis, 3])+0.5
        self.basis_color = nn.Parameter(self.basis_color, requires_grad=False)
        self.original_color = nn.Parameter(self.basis_color, requires_grad=False)

        # extra state for cuda raymarching
        self.cuda_ray = cuda_ray
        if cuda_ray:
            # density grid
            density_grid = torch.zeros([self.cascade, self.grid_size ** 3]) # [CAS, H * H * H]
            density_bitfield = torch.zeros(self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8) # [CAS * H * H * H // 8]
            self.register_buffer('density_grid', density_grid)
            self.register_buffer('density_bitfield', density_bitfield)
            self.mean_density = 0
            self.iter_density = 0
            # step counter
            step_counter = torch.zeros(16, 2, dtype=torch.int32) # 16 is hardcoded for averaging...
            self.register_buffer('step_counter', step_counter)
            self.mean_count = 0
            self.local_step = 0

    def forward(self, x, d):
        raise NotImplementedError()

    # separated density and color query (can accelerate non-cuda-ray mode.)
    def density(self, x):
        raise NotImplementedError()

    def color(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()
        
    def soft2hard(self,a):
        """traslate soft labels to hard labels
            input: a: (batch_size, num_classes): e.g. [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]]
            output: res: (batch_size, num_classes): e.g. [[0, 0, 1], [0, 1, 0]]
        """
        res = torch.zeros_like(a).to(a.device)
        max_idxs = torch.argmax(a, dim=-1)
        batch_iter = torch.arange(a.size(0))
        res[batch_iter, max_idxs] = 1
        return res
    
    def keepmax(self,a):
        """keep the max value and set others to zero
            input: a: (batch_size, num_classes): e.g. [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]]
            output: res: (batch_size, num_classes): e.g. [[0, 0, 0.7], [0, 0.4, 0]]
        """
        res = torch.zeros_like(a)
        max_idxs = torch.argmax(a, dim=-1)
        batch_iter = torch.arange(a.size(0))
        res[batch_iter, max_idxs] = a[batch_iter, max_idxs]
        return res
    
    def reset_extra_state(self):#* will not use, actuallym we have load it when loading the palette checkpoint
        if not self.cuda_ray:
            return 
        # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0
        # step counter
        self.step_counter.zero_()
        self.mean_count = 0
        self.local_step = 0
        
    def run_cuda_seg_image(self, rays_o, rays_d, dt_gamma=0, perturb=False, max_steps=1024, T_thresh=1e-4, white_bg=False, bg_color=None):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device
        #* when use model.train(), self.training will be set to True
        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer, self.min_near)

        #* need: segment information (omega), h_stu_color, density, GT color (diffuse,  offsets)
        #* not need: view_dep, 
        counter = self.step_counter[self.local_step % 16]
        counter.zero_() # set to 0
        self.local_step += 1
        
        dtype = torch.float32
        
        # results = [{'styleIdx':styleIdx,
        #             'color':self.stylizer.new_rgbs[styleIdx],
        #             'rgbs':None,
        #             'image':None,
        #             'style':None} for styleIdx in range(self.num_basis)]
        # n_alive = N
        # rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
        # rays_t = nears.clone() # [N]
        # basis_weights_sum = [torch.zeros(N, dtype=dtype, device=device) for _ in range(self.opt.num_basis)] 
        # basis_depths_sum = [torch.zeros(N, dtype=dtype, device=device) for _ in range(self.opt.num_basis)]
        #todo: maybe we need to implement segment_res() here to avoid frequently copy and paste
        basis_images_output = [torch.zeros(N, 3, dtype=dtype, device=device) for _ in range(self.opt.num_basis)] # each classes images
        basis_images_content = [torch.zeros(N, 3, dtype=dtype, device=device) for _ in range(self.opt.num_basis)] # each classes images
        #* render image for style loss
        basis_weights_sum,basis_masks,basis_images_output = self.segment_res(rays_o, rays_d, nears, fars, dt_gamma=dt_gamma, perturb=perturb, max_steps=max_steps,T_thresh=T_thresh, bg_color=bg_color)
        _, _, basis_images_content = self.segment_res(rays_o, rays_d, nears, fars, dt_gamma=dt_gamma, perturb=perturb, max_steps=max_steps,T_thresh=T_thresh, bg_color=bg_color, stu_color=False)
        # todo: debug check the basis_images here
        # for i in range(self.opt.num_basis):
        #     ic(basis_images[i].shape,basis_masks[i].shape)
        #     plt.figure()
        #     plt.imshow(basis_images[i].reshape(800,800,3).cpu().numpy())n
        #     plt.show()
        #     ic(basis_masks[i]) #visualize the mask
        # exit()    
        # preview basis_weiths_sum[0]
        
        
        return basis_weights_sum,basis_masks,basis_images_output,basis_images_content
            
 
    
    def run_cuda_style(self, rays_o, rays_d, dt_gamma=0, bg_color=None, perturb=False, force_all_rays=False, 
                 max_steps=1024, T_thresh=1e-4, gui_mode=False, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]
        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device
        #* when use model.train(), self.training will be set to True
        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer, self.min_near)

        results = {}
        if bg_color is None:
            bg_color = 1
      
        
        if (self.training):
            results = [{"pred_rgb":None,"tea_color":None,"stu_color":None} for _ in range(self.opt.num_basis)]
            counter = self.step_counter[self.local_step % 16]
            counter.zero_() # set to 0
            self.local_step += 1
            xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, counter, self.mean_count, perturb, 128, force_all_rays, dt_gamma, max_steps)
            M = xyzs.shape[0]
            # sigmas, _, omega, offsets_radiance, view_dep, diffuse, h_stu_color = self(xyzs, dirs)
            sigmas, _, omega, offsets_radiance, view_dep, h_stu_color = self(xyzs, dirs)
            offsets, radiance = offsets_radiance[...,:-1], offsets_radiance[...,-1:]
            sigmas = self.density_scale * sigmas
            sigmas = sigmas.detach()
            segment_masks = self.soft2hard(omega) # segment mask
                        
            radiance = radiance.reshape(M, 1, 1)
            offsets = offsets.reshape(M, self.num_basis, 3)
            omega = omega.reshape(M, self.num_basis, 1)
            view_dep = view_dep.reshape(M, 3)
            # diffuse = diffuse.reshape(M, 3)
            h_stu_color = h_stu_color.reshape(M, 3) #* add
                
            basis_color = self.basis_color[None,:,:].clamp(0, 1)
            # Compositing palette basis
            final_color = (F.softplus(radiance)*(basis_color+offsets))
            basis_rgb = omega*final_color # (N_rays, N_sample, N_basis, 3) 
            
            for i in range(self.opt.num_basis):
                # ic(h_stu_color.shape,segment_masks[:,i].repeat().shape)
                segment_mask = segment_masks[:,i]
            
                segment_mask_color = segment_mask.unsqueeze(-1).expand(-1,3)
            
                masked_sigmas = sigmas*segment_mask
                masked_h_stu_color = h_stu_color*segment_mask_color
                masked_tea_color = basis_rgb[:,i,:]*segment_mask_color
                weights_sum, depth, image = raymarching.composite_rays_train(masked_sigmas, masked_h_stu_color, deltas, rays, T_thresh)
                image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
                # ic(image.requires_grad_())
                # ic(image)
                # exit()
                results[i]["pred_rgb"] = image
                results[i]["tea_color"] = masked_tea_color
                results[i]["stu_color"] = masked_h_stu_color
        else: 
            # allocate outputs 
            # if use autocast, must init as half so it won't be autocasted and lose reference.
            #dtype = torch.half if torch.is_autocast_enabled() else torch.float32
            # output should always be float32! only network inference uses half.
            dtype = torch.float32

            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)
            
            view_dep_rgb_map = torch.zeros(N, 3, dtype=dtype, device=device)
            direct_rgb_map = torch.zeros(N, 3, dtype=dtype, device=device)
            basis_rgb_map = torch.zeros(N, 3*self.opt.num_basis, dtype=dtype, device=device)
            unscaled_basis_rgb_map = torch.zeros(N, 3*self.opt.num_basis, dtype=dtype, device=device)
            basis_acc_map = torch.zeros(N, self.opt.num_basis, dtype=dtype, device=device)
            
            # ic(N) # 640000 = 800*800
            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N] pixel-wise
            rays_t = nears.clone() # [N]
         
            step = 0
            # exit()
            while step < max_steps:

                # count alive rays 
                n_alive = rays_alive.shape[0]
                
                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)# for each ray, at least 1 step, at most 8 steps

                xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, 128, perturb if step == 0 else False, dt_gamma, max_steps)
                M = xyzs.shape[0]
      
                # sigmas, clip_feat, omega, offsets_radiance, view_dep, diffuse, h_stu_color = self(xyzs, dirs, soft2hard=True)
                sigmas, clip_feat, omega, offsets_radiance, view_dep, h_stu_color = self(xyzs, dirs, soft2hard=True)
                offsets, radiance = offsets_radiance[...,:-1], offsets_radiance[...,-1:]
                
                radiance = radiance.reshape(M, 1, 1)
                offsets = offsets.reshape(M, self.num_basis, 3)
                omega = omega.reshape(M, self.num_basis, 1)
                view_dep = view_dep.reshape(M, 3)
                # diffuse = diffuse.reshape(M, 3)
                clip_feat = clip_feat.reshape(M, self.opt.clip_dim)                        
                h_stu_color = h_stu_color.reshape(M, 3) #* add
                basis_color = self.basis_color[None,:,:].clamp(0, 1)
                
                #* photorealistic style transfer
                sigmas = self.density_scale * sigmas
                if self.opt.withoutLighting:
                    rgbs = h_stu_color
                else:
                    rgbs = h_stu_color + view_dep # (N_rays, N_samples_, 3)
                
                ### !!! IMPORTANT !!! make sure this composite rays function is executed after all composite rays flex operations
                # Since this step will modify rays_alive
                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image, T_thresh)

                rays_alive = rays_alive[rays_alive >= 0]

                step += n_step
            # ic(image,bg_color,weights_sum)
            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            depth_origin = depth.clone()
            depth = torch.clamp(depth - nears, min=0) / (fars - nears)
            image = image.view(*prefix, 3)
            depth = depth.view(*prefix)
            depth_origin = depth_origin.view(*prefix)
            results['depth'] = depth
            results['depth_origin'] = depth_origin
            results['image'] = image
            results['weights_sum'] = weights_sum


            if not gui_mode:
                direct_rgb_map = direct_rgb_map + (1 - weights_sum).unsqueeze(-1) * bg_color
                view_dep_rgb_map = view_dep_rgb_map.view(*prefix, 3)
                direct_rgb_map = direct_rgb_map.view(*prefix, 3)
                basis_acc_map = basis_acc_map.view(*prefix, self.num_basis)
                basis_rgb_map = basis_rgb_map.view(*prefix, self.num_basis*3)
                unscaled_basis_rgb_map = unscaled_basis_rgb_map.view(*prefix, self.num_basis*3)
                results['direct_rgb'] = direct_rgb_map
                results['view_dep_rgb'] = view_dep_rgb_map
                results['basis_rgb'] = basis_rgb_map
                results['unscaled_basis_rgb'] = unscaled_basis_rgb_map
                results['basis_acc'] = basis_acc_map
                
        # results['depth'] = depth
        # results['image'] = image
        # results['rgb_norm'] = rgb_norm_map
        # results['weights_sum'] = weights_sum


        return results
    
    def segment_res(self,rays_o, rays_d, nears, fars, dt_gamma=0, perturb=False,
                 max_steps=1024,T_thresh=1e-4, white_bg=True, stu_color=True, bg_color=0):
        dtype = torch.float32
        
        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device
        
        basis_weights_sum = [torch.zeros(N, dtype=dtype, device=device) for _ in range(self.opt.num_basis)]
        basis_depths_sum = [torch.zeros(N, dtype=dtype, device=device) for _ in range(self.opt.num_basis)] # not used
        basis_images = [torch.zeros(N, 3, dtype=dtype, device=device) for _ in range(self.opt.num_basis)]
        basis_masks = [torch.zeros(N, dtype=dtype, device=device) for _ in range(self.opt.num_basis)]
        
        n_alive = N
        basis_rays_alive = [torch.arange(n_alive, dtype=torch.int32, device=device) for _ in range(self.opt.num_basis)] # [N]
        
        
        
        for i in range(self.opt.num_basis-1,-1,-1):
            rays_alive = basis_rays_alive[i]
            step = 0
            rays_t = nears.clone() # [N]
            n_alive = N
            while step < max_steps:
                # count alive rays 
                n_alive = rays_alive.shape[0]
                
                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)
                xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, 128, perturb if step == 0 else False, dt_gamma, max_steps)
                M = xyzs.shape[0]
                # sigmas, _, omega, offsets_radiance, view_dep, diffuse, h_stu_color = self(xyzs, dirs)
                sigmas, _, omega, offsets_radiance, view_dep, h_stu_color = self(xyzs, dirs)
                offsets, radiance = offsets_radiance[...,:-1], offsets_radiance[...,-1:]
                
                radiance = radiance.reshape(M, 1, 1)
                offsets = offsets.reshape(M, self.num_basis, 3)
                segment_masks = self.soft2hard(omega)
                omega = omega.reshape(M, self.num_basis, 1)
                view_dep = view_dep.reshape(M, 3)
                basis_color = self.basis_color[None,:,:].clamp(0, 1)   # [1,2,3]
                final_color = (F.softplus(radiance)*(basis_color+offsets))# .clamp(0, 1)
                if stu_color: 
                    basis_rgb = h_stu_color# +  view_dep.detach() 
                else:#* if we want to generate content image, we should use the teacher's color
                    basis_rgb = (omega*final_color).sum(dim=-2).detach()# +  view_dep.detach()
                sigmas = self.density_scale * sigmas     
                masked_sigmas = sigmas.detach()*segment_masks[:,i]
                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, masked_sigmas, basis_rgb, deltas, basis_weights_sum[i], basis_depths_sum[i], basis_images[i], T_thresh)             
                rays_alive = rays_alive[rays_alive >= 0]
                step += n_step
            basis_masks[i] = torch.where(basis_weights_sum[i] != 0.0, 1.0,0.0)
            # if white_bg:
            basis_images[i] = basis_images[i] + (1 - basis_weights_sum[i]).unsqueeze(-1) * bg_color[i]
            
        return basis_weights_sum,basis_masks,basis_images

    

    def run_cuda_photo(self, rays_o, rays_d, dt_gamma=0, bg_color=None, perturb=False, force_all_rays=False, 
                 max_steps=1024, T_thresh=1e-4, gui_mode=False, stu_color=True, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]
        if not isinstance(stu_color, list):
            stu_color = [stu_color for _ in range(self.opt.num_basis)]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device
        #* when use model.train(), self.training will be set to True
        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer, self.min_near)

        # mix background color
        if self.bg_radius > 0:
            # use the bg model to calculate bg_color
            sph = raymarching.sph_from_ray(rays_o, rays_d, self.bg_radius) # [N, 2] in [-1, 1]
            bg_color = self.background(sph, rays_d) # [N, 3]
        elif bg_color is None:
            bg_color = 1

        results = {}
        # ic(self.training) #gui mode: False
        if self.training:
            # setup counter
            counter = self.step_counter[self.local_step % 16]
            counter.zero_() # set to 0
            self.local_step += 1
            xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, counter, self.mean_count, perturb, 128, force_all_rays, dt_gamma, max_steps)
            M = xyzs.shape[0]
            # Predict sigma, feature and palette basis from network
            # sigmas, clip_feat, omega, offsets_radiance, view_dep, diffuse, h_stu_color = self(xyzs, dirs)
            sigmas, clip_feat, omega, offsets_radiance, view_dep, h_stu_color = self(xyzs, dirs)
            offsets, radiance = offsets_radiance[...,:-1], offsets_radiance[...,-1:]
            sigmas = self.density_scale * sigmas
            sigmas = sigmas.detach()
            #* photorealistic style transfer
            radiance = radiance.reshape(M, 1, 1)
            offsets = offsets.reshape(M, self.num_basis, 3)
            omega = omega.reshape(M, self.num_basis, 1)
            view_dep = view_dep.reshape(M, 3)
            # diffuse = diffuse.reshape(M, 3)
            h_stu_color = h_stu_color.reshape(M, 3) #* add
            clip_feat = clip_feat.reshape(M, self.opt.clip_dim)

            basis_color = self.basis_color[None,:,:].clamp(0, 1)
            if self.freeze_basis_color:
                basis_color = basis_color.detach()
            final_color = (F.softplus(radiance)*(basis_color+offsets)) 
            basis_rgb = omega*final_color # (N_rays, N_sample, N_basis, 3) 
            tea_rgbs = basis_rgb.sum(dim=-2) #* distill target, when palette is changed
            
            results['stu_rgbs'] = h_stu_color
            results['tea_rgbs'] = tea_rgbs 
            return results
        else: #todo: modify test branch
            # allocate outputs 
            # if use autocast, must init as half so it won't be autocasted and lose reference.
            #dtype = torch.half if torch.is_autocast_enabled() else torch.float32
            # output should always be float32! only network inference uses half.
            dtype = torch.float32

            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)
            
            view_dep_rgb_map = torch.zeros(N, 3, dtype=dtype, device=device)
            direct_rgb_map = torch.zeros(N, 3, dtype=dtype, device=device)
            basis_rgb_map = torch.zeros(N, 3*self.opt.num_basis, dtype=dtype, device=device)
            unscaled_basis_rgb_map = torch.zeros(N, 3*self.opt.num_basis, dtype=dtype, device=device)
            basis_acc_map = torch.zeros(N, self.opt.num_basis, dtype=dtype, device=device)
            
            # ic(N) # 640000 = 800*800
            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N] pixel-wise
            rays_t = nears.clone() # [N]
         
            step = 0
            # exit()
            while step < max_steps:

                # count alive rays 
                n_alive = rays_alive.shape[0]
                
                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)# for each ray, at least 1 step, at most 8 steps

                xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, 128, perturb if step == 0 else False, dt_gamma, max_steps)
                M = xyzs.shape[0]
      
                # sigmas, clip_feat, omega, offsets_radiance, view_dep, diffuse, h_stu_color = self(xyzs, dirs, soft2hard=True)
                sigmas, clip_feat, omega, offsets_radiance, view_dep, h_stu_color = self(xyzs, dirs, soft2hard=True)
                offsets, radiance = offsets_radiance[...,:-1], offsets_radiance[...,-1:]
                
                radiance = radiance.reshape(M, 1, 1)
                offsets = offsets.reshape(M, self.num_basis, 3)
                segment_masks = self.soft2hard(omega)
                # ic(segment_masks.shape)
                # ic(sigmas[segment_masks[:,0]==1].shape)
                # ic(sigmas.shape)
                # exit()
                
                omega = omega.reshape(M, self.num_basis, 1)
                view_dep = view_dep.reshape(M, 3)
                # diffuse = diffuse.reshape(M, 3)
                clip_feat = clip_feat.reshape(M, self.opt.clip_dim)                        
                h_stu_color = h_stu_color.reshape(M, 3) #* add
                basis_color = self.basis_color[None,:,:].clamp(0, 1)
                
                #* photorealistic style transfer
                sigmas = self.density_scale * sigmas
                rgbs_buffer = torch.zeros_like(h_stu_color)
                final_color = (F.softplus(radiance)*(basis_color+offsets))
                basis_rgb = (omega*final_color).sum(dim=-2).float() # (N_rays, N_samples_, 3)
                # ic(basis_rgb.shape) # 640128 3
                # ic(h_stu_color.shape) # 640128 3
                # ic(sigmas.shape) # 640128
                # ic(segment_masks.shape) # 640128 2
                # exit()
                for i in range(self.opt.num_basis): #* adjusting the density, offset and view_dep based on segment result
                    sigmas[segment_masks[:,i]==1] *= self.density_weight[i] 
                    # offsets[segment_masks[:,i]==1] *= self.offsets_weight[i]
                    view_dep[segment_masks[:,i]==1] *= self.view_dep_weight[i]
                    if stu_color[i]: # use student color for palette i
                        rgbs_buffer[segment_masks[:,i]==1] = h_stu_color[segment_masks[:,i]==1]
                    else:
                        rgbs_buffer[segment_masks[:,i]==1] = basis_rgb[segment_masks[:,i]==1]
                    # rbgs_buffer[segment_masks[:,i]==1] = h_stu_color[segment_masks[:,i]==1]
                    # rbgs_buffer[segment_masks[:,i]==0] = basis_rgb[segment_masks[:,i]==0]
                
                
                rgbs = rgbs_buffer + view_dep # (N_rays, N_samples_, 3)
                # if stu_color:
                #     rgbs = h_stu_color + view_dep # (N_rays, N_samples_, 3)
                # else:
                #     # offsets = torch.rand_like(offsets)
                #     # final_color = (F.softplus(radiance)*(basis_color+offsets))#+offset# .clamp(0, 1)
                #     final_color = (F.softplus(radiance)*(basis_color+offsets))
                #     basis_rgb = omega*final_color
                #     rgbs = basis_rgb.sum(dim=-2) + view_dep

                ### !!! IMPORTANT !!! make sure this composite rays function is executed after all composite rays flex operations
                # !!! Since this step will modify rays_alive
                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image, T_thresh)

                rays_alive = rays_alive[rays_alive >= 0]

                step += n_step
            
            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            depth_origin = depth.clone()
            depth = torch.clamp(depth - nears, min=0) / (fars - nears)
            image = image.view(*prefix, 3)
            depth = depth.view(*prefix)
            depth_origin = depth_origin.view(*prefix)
            results['depth'] = depth
            results['depth_origin'] = depth_origin
            results['image'] = image
            results['weights_sum'] = weights_sum


            if not gui_mode:
                direct_rgb_map = direct_rgb_map + (1 - weights_sum).unsqueeze(-1) * bg_color
                view_dep_rgb_map = view_dep_rgb_map.view(*prefix, 3)
                direct_rgb_map = direct_rgb_map.view(*prefix, 3)
                basis_acc_map = basis_acc_map.view(*prefix, self.num_basis)
                basis_rgb_map = basis_rgb_map.view(*prefix, self.num_basis*3)
                unscaled_basis_rgb_map = unscaled_basis_rgb_map.view(*prefix, self.num_basis*3)
                results['direct_rgb'] = direct_rgb_map
                results['view_dep_rgb'] = view_dep_rgb_map
                results['basis_rgb'] = basis_rgb_map
                results['unscaled_basis_rgb'] = unscaled_basis_rgb_map
                results['basis_acc'] = basis_acc_map

        return results
    
    #* ======================================================================================================= *#
    
    
    def render(self, rays_o, rays_d, staged=False, max_ray_batch=4096, test_mode=False, gui_mode=False, color_segment=False, stu_color=True, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]
        # self.photo_mode = False #todo: for debug use
        if self.photo_mode: 
            _run = self.run_cuda_photo
        else:
            _run = self.run_cuda_style
        # B, N = rays_o.shape[:2]
        # device = rays_o.device

        results = _run(rays_o, rays_d, gui_mode=gui_mode, color_segment=color_segment, stu_color=stu_color, **kwargs)

        return results
    




        
