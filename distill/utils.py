import os
import glob
import tqdm
import math
import imageio
import random
import warnings
import tensorboardX

import numpy as np
import pandas as pd

import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from skimage import io, color

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.utils.data import Dataset, DataLoader

import trimesh
import mcubes
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver
import palette
from nerf.utils import *
import lpips

from typing import Tuple
from .rgbsg import *
import copy
from .nn_loss import *
from sklearn.decomposition import PCA

try:
    import _palette_func as _backend
except ImportError:
    print("Loading module palette_func...")
    from .backend import _backend

class SparsityMeter:
    def __init__(self, opt, device=None):
        self.V = 0
        self.N = 0
        self.num_basis = opt.num_basis
        self.device=device
        self.basis_metric=True

    def clear(self):
        self.V = 0
        self.N = 0
        
    def update(self, omega):
        omega = omega.to(self.device) # [B, H, W, N_p], range[0, 1]
        
        # simplified since max_pixel_value is 1 here.
        # psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        omega_sparsity = omega.sum(dim=-1, keepdim=True)/((omega**2).sum(dim=-1, keepdim=True)+1e-6)-1 # N_rays, N_sample, 1
        
        self.V += omega_sparsity.mean()
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "Sparsity"), self.measure(), global_step)

    def report(self):
        return f'Sparsity = {self.measure():.6f}'

class TVMeter:
    def __init__(self, opt, device=None):
        self.V = 0
        self.N = 0
        self.num_basis = opt.num_basis
        self.device=device
        self.basis_metric=True

    def clear(self):
        self.V = 0
        self.N = 0
        
    def update(self, omega):
        omega = omega.to(self.device) # [B, H, W, N_p], range[0, 1]
        
        # simplified since max_pixel_value is 1 here.
        # psnr = -10 * np.log10(np.mean((preds - truths) ** 2))

        w_variance = torch.mean(torch.pow(omega[:,:,:-1,:] - omega[:,:,1:,:], 2))        
        h_variance = torch.mean(torch.pow(omega[:,:-1,:,:] - omega[:,1:,:,:], 2))
        tv = w_variance + h_variance
        self.V += tv*100
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "TV"), self.measure(), global_step)

    def report(self):
        return f'TV = {self.measure():.6f}'

    
def get_palette_weight_with_hist(rgb, hist_weights):
    assert(hist_weights.ndim == 5)
    rgb_shape = rgb.shape
    rgb = rgb.reshape(-1, 3)
    rgb = rgb[None,None,None,:,[2,1,0]]*2-1
    weight = torch.nn.functional.grid_sample(hist_weights, rgb, mode='bilinear', padding_mode='zeros', align_corners=True)
    weight = weight.squeeze().permute(1, 0)
    return weight.reshape(rgb_shape[:-1] + (-1,))

def normalize(tensor):
    return tensor / (tensor.norm(dim=-1, keepdim=True)+1e-9)
   
def compute_RGB_histogram(
    colors_rgb: np.ndarray,
    weights: np.ndarray,
    bits_per_channel: int
) -> Tuple[np.ndarray, np.ndarray]:
    assert colors_rgb.ndim == 2 and colors_rgb.shape[1] == 3
    assert weights.ndim == 1
    assert len(colors_rgb) == len(weights)
    assert 1 <= bits_per_channel and bits_per_channel <=8

    try:
        bin_weights, bin_centers_rgb = _backend.compute_RGB_histogram(
            colors_rgb.flatten(), weights.flatten(), bits_per_channel)
    except RuntimeError as err:
        print(err)
        assert False

    return bin_weights, bin_centers_rgb

def run_kmeans(
    n_clusters: int,
    points: np.ndarray,
    init: np.ndarray,
    sample_weight: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    print(f'running kmeans with K = {n_clusters}')
    kmeans = KMeans(n_clusters=n_clusters, init=init).fit(X=points, sample_weight=sample_weight)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    center_weights = np.zeros(n_clusters)
    for i in range(n_clusters):
        center_weights[i] = np.sum(sample_weight[labels==i])

    idcs = np.argsort(center_weights * -1)

    return centers[idcs], center_weights[idcs]

def palette_extraction(
    inputs: dict,
    output_dir: str,
    tau: float = 8e-3,
    palette_size = None,
    normalize_input = False,
    error_thres = 5.0 / 255.0
):
    '''
        Extract palettes with the RGBXY method
    '''
    assert palette_size is None or palette_size >= 4 ## convex hull should have at least 4 vertices
    print(f'extracting palette with {palette_size} colors')

    if not os.path.exists(output_dir):
        print(f'create output directory {output_dir}')
        os.makedirs(output_dir)
    
    output_prefix = "%s/extract"%output_dir
    ## radiance sampling
    start = time.time()

    colors = inputs['colors']
    weights = np.ones_like(colors[...,0])
    colors = colors.reshape(-1,3)
    weights = weights.flatten()

    assert len(weights[weights < 0]) == 0, 'negative weight indicates the failure of radiance sampling'

    ## save radiance samples (outside timing analysis)
    res = 800
    n_total = res**2
    random.seed(0)
    idcs = random.sample(range(len(colors)), n_total)
    assert len(idcs) == len(set(idcs)), 'each element of idcs should be unique'
    img = colors[idcs].reshape(res,res,3)
    Image.fromarray((img*255).round().clip(0,255).astype(np.uint8)).save(output_prefix+"-radiance-raw.png")

    ## radiance sample filtering
    start = time.time()

    ## coarse histogram (2^3 = 8 bins)
    bin_weights_coarse, bin_centers_coarse = compute_RGB_histogram(colors, weights, bits_per_channel=3)
    sum_weights = np.sum(bin_weights_coarse)
    bin_weights_coarse /= sum_weights

    idcs = bin_weights_coarse > tau
    bin_weights_coarse = bin_weights_coarse[idcs]
    bin_centers_coarse = bin_centers_coarse[idcs]

    ## fine histogram (2^5 = 32 bins)
    bin_weights_fine, bin_centers_fine = compute_RGB_histogram(colors, weights, bits_per_channel=5)
    idcs = bin_weights_fine > 0
    bin_weights_fine = bin_weights_fine[idcs]
    bin_weights_fine /= sum_weights
    bin_centers_fine = bin_centers_fine[idcs]

    centers, center_weights = run_kmeans(
        n_clusters=len(bin_weights_coarse), points=bin_centers_fine,
        init=bin_centers_coarse, sample_weight=bin_weights_fine)

    ## convex hull simplification
    start = time.time()
    palette_rgb = Hull_Simplification_posternerf(
        centers.astype(np.double), output_prefix,
        pixel_counts=center_weights,
        error_thres=error_thres,
        target_size=palette_size)
    _, hist_rgb = compute_RGB_histogram(colors, weights, bits_per_channel=5)
    
    if normalize_input:
        hist_rgb = hist_rgb + 0.05
        hist_rgb_norm = np.linalg.norm(hist_rgb, axis=-1, keepdims=True) #.clip(min=0.1)
        hist_rgb = hist_rgb / hist_rgb_norm

    # Generate weight
    hist_weights = Tan18.Get_ASAP_weights_using_Tan_2016_triangulation_and_then_barycentric_coordinates(hist_rgb.astype(np.double).reshape((-1,1,3)), 
                        palette_rgb, None, order=0) # N_bin_center x 1 x num_palette
    hist_weights = hist_weights.reshape([32,32,32,palette_rgb.shape[0]])
    

    num_colors = palette_rgb.shape[0]
    sel_color_idx = []
    new_palette_rgb = []
    #*KT： remove similiar color
    for color_idx in range(num_colors): # traverse all colors
        if not any(np.array_equal(palette_rgb[color_idx], x) for x in new_palette_rgb): # if not in new palette
            if new_palette_rgb == []: # if new palette is empty
                new_palette_rgb.append(palette_rgb[color_idx])
                sel_color_idx.append(color_idx)
            else: # if new palette is not empty
                has_similiar = False
                for new_color_idx in range(len(new_palette_rgb)): # traverse new palette, compare with each color
                    dist = np.dot(palette_rgb[color_idx], new_palette_rgb[new_color_idx]) / (np.linalg.norm(palette_rgb[color_idx]) * np.linalg.norm(new_palette_rgb[new_color_idx]))
                    if dist > 0.8: # there is a similiar color already in new_palette, do not add
                        has_similiar = True
                        break
                if not has_similiar:
                    new_palette_rgb.append(palette_rgb[color_idx]) # if no similiar color, add to new palette
                    sel_color_idx.append(color_idx)
    palette_rgb = np.array(new_palette_rgb)
    hist_weights = hist_weights[...,sel_color_idx]

    ## save palette
    palette_img = get_bigger_palette_to_show(palette_rgb)
    Image.fromarray((palette_img*255).round().clip(0,255).astype(np.uint8)).save(output_prefix+"-palette.png")

    write_palette_txt(palette_rgb, output_prefix+'-palette.txt')
    np.savez(os.path.join(output_dir, 'palette.npz'), palette=palette_rgb)
    np.savez(os.path.join(output_dir, 'hist_weights.npz'), hist_weights=hist_weights)

# CUDA-based rgb,hsv conversion for speed-up
class _rgb_to_hsv(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input):

        if not input.is_cuda: input = input.cuda()
        
        prefix = input.shape[:-1]
        input = input.contiguous().view(-1, 3)

        n_rays = input.shape[0]
        output = torch.empty(n_rays, 3, device=input.device, dtype=input.dtype)

        _backend.rgb_to_hsv(n_rays, input, output)
        output = output.reshape(*prefix, 3)

        return output

rgb_to_hsv = _rgb_to_hsv.apply

class _hsv_to_rgb(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input):

        if not input.is_cuda: input = input.cuda()
        
        prefix = input.shape[:-1]
        input = input.contiguous().view(-1, 3)

        n_rays = input.shape[0]
        output = torch.empty(n_rays, 3, device=input.device, dtype=input.dtype)

        _backend.hsv_to_rgb(n_rays, input, output)
        output = output.reshape(*prefix, 3)

        return output

hsv_to_rgb = _hsv_to_rgb.apply

def nerf_matrix_to_ngp(pose, scale=0.8):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array(
        [
            [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
            [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
            [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    return new_pose

def pose_spherical(theta, phi, radius):
    # for synthetic. it generates sphere random poses
    trans_t = lambda t: np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]
    ).astype(np.float32)
    rot_phi = lambda phi: np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ]
    ).astype(np.float32)
    rot_theta = lambda th: np.array(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ]
    ).astype(np.float32)
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).astype(
            np.float32
        )
        @ c2w
    )
    return c2w

def get_rand_poses(data_type="synthetic", original_loader=None):
    """
    Random sampling. Random origins and directions.
    """
    from scipy.spatial.transform import Slerp, Rotation

    assert data_type in {"synthetic", "llff", "tank"}

    def get_single_syn_pose(ph, rand_radius=False):
        theta1 = -180
        theta2 = 180
        phi1 = -ph
        phi2 = 5 - ph if (5 - ph) <= 0 else 0
        theta = theta1 + np.random.rand() * (theta2 - theta1)
        phi = phi1 + np.random.rand() * (phi2 - phi1)
        if rand_radius:
            radius = np.random.uniform(3, 4)
        else:
            radius = 4
        return pose_spherical(theta, phi, radius)
    

    def get_syn_poses():
        random_poses = np.array([get_single_syn_pose(8) for _ in range(1)])
        for a in range(0, 80):
            rp = np.array(
                [get_single_syn_pose(a) for _ in range(int(((90 - a) // 15) ** 1 + 1))]
            )
            random_poses = np.concatenate([random_poses, rp], axis=0)
        for i in range(len(random_poses)):
            random_poses[i] = nerf_matrix_to_ngp(random_poses[i])
        print(f"\nlen(train data): {len(random_poses)}\n")
        random_poses = torch.from_numpy(random_poses).cuda()
        return random_poses

    def get_tank_poses():
        random_poses = np.array([get_single_syn_pose(8) for _ in range(1)])
        for a in range(5, 20):
            rp = np.array(
                [
                    get_single_syn_pose(a, True)
                    for _ in range(int(((90 - a) // 15) ** 1 + 1))
                ]
            )
            random_poses = np.concatenate([random_poses, rp], axis=0)
        for i in range(len(random_poses)):
            random_poses[i] = nerf_matrix_to_ngp(random_poses[i])
        print(f"\nlen(train data): {len(random_poses)}\n")
        random_poses = torch.from_numpy(random_poses).cuda()
        return random_poses

    def rand_poses_from_cam_centers(centers):
        def normalize(vectors):
            return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

        size = len(centers)
        forward_vector = -normalize(centers)
        up_vector = (
            torch.FloatTensor([0, -1, 0]).to("cuda").unsqueeze(0).repeat(size, 1)
        )  # confused at the coordinate system...
        right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
        up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

        poses = (
            torch.eye(4, dtype=torch.float, device="cuda")
            .unsqueeze(0)
            .repeat(size, 1, 1)
        )
        poses[:, :3, :3] = torch.stack(
            (right_vector, up_vector, forward_vector), dim=-1
        )
        poses[:, :3, 3] = centers
        return poses

    def get_llff_poses_rand():
        def get_rand_cam_centers_from_bbox(poses, gen_num=30):
            # use poses to estimate the bbox of the camera
            trasitions = poses[:, :3, 3]
            bbox_max = trasitions.max(axis=0) + 1e-6
            bbox_min = trasitions.min(axis=0) - 1e-6
            rand_xs = np.random.uniform(low=bbox_min[0], high=bbox_max[0], size=gen_num)
            rand_ys = np.random.uniform(low=bbox_min[1], high=bbox_max[1], size=gen_num)
            rand_zs = np.random.uniform(low=bbox_min[2], high=bbox_max[2], size=gen_num)
            centers = np.stack([rand_xs, rand_ys, rand_zs], axis=1)
            return centers.astype(np.float32)

        centers = get_rand_cam_centers_from_bbox(original_loader)
        random_poses = rand_poses_from_cam_centers(torch.from_numpy(centers).cuda())
        random_poses[:, 0, 0] = -random_poses[:, 0, 0]
        return random_poses

    if data_type == "synthetic":
        random_poses = get_syn_poses()
    elif data_type == "llff":
        random_poses = get_llff_poses_rand()
    elif data_type == "tank":
        random_poses = get_tank_poses()
    else:
        raise ValueError("illegal")
    return random_poses

class StyleRFVolVisTrainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 palette_path=None, #* was nerf_path
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):
        
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.val_len = 1
        self.require_smooth_loss = False
        self.stage1_total_iters = self.opt.stage1_iters
        self.distill_executed = False
        self.style_weight = [1.0 for _ in range(opt.num_basis)]
       

        model.to(self.device)
        
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model
        self.model.photo_mode = True #* explicitly set photo_mode to True

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        # optionally use LPIPS loss for patch-based training
        if self.opt.patch_size > 1:
            import lpips
            self.criterion_lpips = lpips.LPIPS(net='alex').to(self.device)

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init for distill
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }
        
        self.style_epoch = 0
        self.style_global_step = 0
        self.style_local_step = 0

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)
        
        self.palette_path = palette_path
        if self.palette_path is not None:
            self.log(f"[INFO] Loading NeRF at {self.palette_path} ...")
            self.load_palette_checkpoint(self.palette_path) # was load_nerf_checkpoint
             
    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()
            
    def reset_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.opt.lr

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    def eval_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        images = data['images'] # [B, H, W, 3/4]
        B, H, W, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        # eval with fixed background color
        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images
        
        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, test_mode=True, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)

        loss = self.criterion(pred_rgb, gt_rgb).mean()

        return pred_rgb, pred_depth, gt_rgb, loss, outputs
    
    def distill(self, train_loader, valid_loader, max_epochs):
        self.hard_rays_pool = [torch.tensor([]).cuda(), torch.tensor([]).cuda()]
        self.is_hard_rays_pool_full = False

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(
                os.path.join(self.workspace, "run", self.name)
            )

        #* freeze teacher model
        self.model.freeze_teacher() #*KT: freeze teacher model
        self.model.original_color = self.model.basis_color
        self.model.stylizer.update_delta_hsv(rgb_orig=self.model.basis_color) 
        if self.opt.recolor:
            self.model.basis_color = self.model.stylizer.recolor(self.model.basis_color)
        
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(name)
        # exit()
        
        # get a ref to error_map
        #* error map is used to store the error of each pixel, which is used to sample hard pixels
        self.error_map = train_loader._data.error_map

        if (
            not self.opt.use_real_data_for_train
        ):  # using random poses to calculate max_epochs.
            random_poses = get_rand_poses(
                data_type="synthetic",
                original_loader=copy.deepcopy(
                    train_loader._data.poses.detach().cpu().numpy()
                ),
            )
            self.opt.iters = int(
                (self.opt.iters // len(random_poses)) * len(random_poses)
            )
            max_epochs = np.ceil(self.opt.iters / len(random_poses)).astype(np.int32)
            scheduler = lambda optimizer: optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.opt.iters * 1, eta_min=7e-5
            )  # update scheduler according to new opt.iters
            self.lr_scheduler = scheduler(self.optimizer)

        self.total_epoch = max_epochs
        self.log(f"\n----------------total epoch:{max_epochs} -----------\n")

        self.real_train_poses = copy.deepcopy(train_loader._data.poses)
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch
            if not self.opt.use_real_data_for_train:
                print(f"\n generate new random poses at epoch{self.epoch}")
                random_poses = get_rand_poses(
                    data_type="synthetic",
                    original_loader=self.real_train_poses.detach().cpu().numpy(),
                )
                train_loader._data.poses = copy.deepcopy(random_poses)
                train_loader._data.images = train_loader._data.images[:1].expand(
                    len(random_poses), -1, -1, -1
                )
                train_loader = train_loader._data.dataloader()
            self.distill_one_epoch(train_loader)
            print("\n", self.workspace, "\n")

            if (
                self.workspace is not None
                and self.local_rank == 0
                and self.epoch > max_epochs - 1
            ):
                self.save_checkpoint(full=False, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)  

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()
            

    def evaluate(self, loader, name=None, save_images=True):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name, save_images=save_images)
        self.use_tensorboardX = use_tensorboardX
    
    def get_loss(self, pred, gt):
        loss = torch.mean((gt - pred) ** 2)
        return loss
    
    def distill_step(self, data):
        rays_o = data["rays_o"]  # [B, N, 3]
        rays_d = data["rays_d"]  # [B, N, 3]  [1, N=rays_num=4096, 3]
        loss = 0.0
        images = data["images"]  # [B, N, 3/4]
        B, N, C = images.shape
        bg_color = torch.rand(
            [B, rays_o.size(1), 3], dtype=images.dtype, device=images.device
        )
        outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, force_all_rays=True, **vars(self.opt))
        loss_dict = {}
        loss = 0.0
        if (self.global_step < self.opt.stage1_iters
            and self.model.stu_feat is not None
            and self.model.tea_feat is not None):
            loss_feat = self.opt.loss_rate_fea_sc * self.get_loss(self.model.stu_feat, self.model.tea_feat)
            loss += loss_feat
            return loss, loss_dict
        else:
            loss_color = self.get_loss(outputs['stu_rgbs'], outputs['tea_rgbs'])
            loss += loss_color
        return loss, loss_dict

    
    def distill_one_epoch(self, loader):
        self.log(f"==> Start Distillation Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")
        
        total_loss = 0

        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()
        
        self.model.train() # set self.training = True
        
        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                loss, loss_dict = self.distill_step(data) 

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            
        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def evaluate_one_epoch(self, loader, name=None, save_images=True, color_segment=False):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval() # set self.training = False

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        data_len = min(len(loader), self.val_len)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=data_len * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data_idx, data in enumerate(loader):    
                self.local_step += 1
                if self.local_step == data_len+1:
                    break

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, truths, loss, outputs = self.eval_step(data) #todo: key step

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size
                    
                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)
                
                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    for metric in self.metrics:
                        if hasattr(metric, "basis_metric"):
                            basis_acc = outputs['basis_acc'][0].reshape(1, preds[0].shape[0], preds[0].shape[1], self.opt.num_basis)
                            metric.update(basis_acc)
                        else:
                            metric.update(preds, truths)
                        
                    # Render evaluation images/passes
                    if save_images:
                        # save image
                        save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                        save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_depth.png')

                        save_path_basis_color = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_basis_color.png')

                        os.makedirs(os.path.dirname(save_path), exist_ok=True)

                        if self.opt.color_space == 'linear':
                            preds = linear_to_srgb(preds)

                        pred = preds[0].detach().cpu().numpy()
                        pred = (pred.clip(0,1) * 255).astype(np.uint8)
                        pred_depth = preds_depth[0].detach().cpu().numpy()
                        pred_depth = (pred_depth * 255).astype(np.uint8)
                        pred_view_dep_color = outputs['view_dep_rgb'][0].detach().cpu().numpy()
                        pred_view_dep_color = pred_view_dep_color.reshape(pred.shape[0], pred.shape[1], 3)
                        pred_view_dep_color = (pred_view_dep_color.clip(0,1) * 255).astype(np.uint8)
                        pred_direct_color = outputs['direct_rgb'][0].detach().cpu().numpy()
                        pred_direct_color = pred_direct_color.reshape(pred.shape[0], pred.shape[1], 3)
                        pred_direct_color = (pred_direct_color.clip(0,1) * 255).astype(np.uint8)

                        pred_basis_img = []
                        pred_basis_acc = []
                        pred_basis_color = []
                        pred_unscaled_basis_color = []
                        for i in range(self.opt.num_basis):
                            basis_img = outputs['basis_rgb'][0,:,i*3:(i+1)*3].reshape(pred.shape[0], pred.shape[1], 3)
                            pred_basis_img.append(basis_img.detach().cpu().numpy())  

                            unscaled_basis_img = outputs['unscaled_basis_rgb'][0,:,i*3:(i+1)*3].reshape(pred.shape[0], pred.shape[1], 3)
                            pred_unscaled_basis_color.append(unscaled_basis_img.detach().cpu().numpy())  

                            basis_acc = outputs['basis_acc'][0,:,i:(i+1)].reshape(pred.shape[0], pred.shape[1])
                            pred_basis_acc.append(basis_acc.detach().cpu().numpy())

                            basis_color = self.model.basis_color[i,None,None,:].repeat(100, 100, 1)
                            basis_color = basis_color.clamp(0, 1)
                            pred_basis_color.append(basis_color.detach().cpu().numpy())

                        pred_basis_img = (np.concatenate(pred_basis_img, axis=1).clip(0,1)*255).astype(np.uint8)
                        pred_basis_acc = (np.concatenate(pred_basis_acc, axis=1).clip(0,1)*255).astype(np.uint8)
                        pred_basis_color = (np.concatenate(pred_basis_color, axis=1).clip(0,1)*255).astype(np.uint8)
                        pred_unscaled_basis_color = (np.concatenate(pred_unscaled_basis_color, axis=1).clip(0,1)*255).astype(np.uint8)

                        # cv2.imwrite(save_path_basis_img, cv2.cvtColor(pred_basis_img, cv2.COLOR_RGB2BGR))
                        # cv2.imwrite(save_path_basis_acc, pred_basis_acc)
                        cv2.imwrite(save_path_basis_color, cv2.cvtColor(pred_basis_color, cv2.COLOR_RGB2BGR))
                        # cv2.imwrite(save_path_unscaled_basis_color, cv2.cvtColor(pred_unscaled_basis_color, cv2.COLOR_RGB2BGR))
                        # cv2.imwrite(save_path_view_dep_color, cv2.cvtColor(pred_view_dep_color, cv2.COLOR_RGB2BGR))
                        
                        cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(save_path_depth, pred_depth)

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)


        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False, gui_mode=False, stu_color=False):  

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=perturb, gui_mode=gui_mode, stu_color=stu_color, test_mode=True, **vars(self.opt))

        outputs['preds'] = outputs['image']
        outputs['preds_depth'] = outputs['depth']
        if 'depth_origin' not in outputs.keys():
            outputs['depth_origin'] = outputs['depth']
        outputs['preds_xyz'] = data['rays_o'] + data['rays_d'] * outputs['depth_origin'][...,None]
        if 'weights_sum' in outputs.keys():
            outputs['preds_weight'] = outputs['weights_sum']

        return outputs 

    def test(self, loader, save_path=None, name=None, write_video=True, selected_idx=None, gui_mode=False, eval_time=False):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, 'imgs'), exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []
            all_preds_basis_img = []
            all_preds_basis_acc = []
            all_preds_basis_color = []
            all_preds_view_dep_color = []

        with torch.no_grad():
            tic=torch.cuda.Event(enable_timing=True)
            toc=torch.cuda.Event(enable_timing=True)
            tic.record()
            # import time
            # tik = time.time()
            for t, data in enumerate(loader):
                if selected_idx is not None and t != selected_idx:
                    continue
                H, W = data['H'], data['W']
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    outputs = self.test_step(data, gui_mode=gui_mode)


                pred = outputs['preds'][0].reshape(H, W, 3)
                
                if self.opt.color_space == 'linear':
                    pred = linear_to_srgb(pred)
                    
                pred = pred.detach().cpu().numpy()
                pred = (pred.clip(0, 1) * 255).astype(np.uint8)

                pred_depth = outputs['preds_depth'][0].reshape(H, W).detach().cpu().numpy()
                pred_depth = (pred_depth * 255).astype(np.uint8)
                
                # prepare results of all testing frames
                if not self.opt.gui:
                    pred_view_dep_color = outputs['view_dep_rgb'][0].reshape(H, W, 3)
                    pred_view_dep_color = pred_view_dep_color.detach().cpu().numpy()
                    pred_view_dep_color = (pred_view_dep_color.clip(0, 1) * 255).astype(np.uint8)

                    pred_basis_img = []
                    pred_basis_acc = []
                    pred_basis_color = []

                    for i in range(self.opt.num_basis):
                        basis_img = outputs['basis_rgb'][0,:,i*3:(i+1)*3].reshape(H, W, 3)
                        pred_basis_img.append(basis_img.detach().cpu().numpy())                        
                        basis_acc = outputs['basis_acc'][0,:,i:(i+1)].reshape(H, W)
                        pred_basis_acc.append(basis_acc.detach().cpu().numpy())
                        basis_color = self.model.basis_color[i,None,None,:].repeat(100, 100, 1)
                        basis_color = basis_color.clamp(0, 1)
                        # basis_color = lab_to_rgb(torch.concatenate([basis_color[...,:1]*0+75, basis_color[...,1:]], dim=-1))
                        pred_basis_color.append(basis_color.detach().cpu().numpy())

                    pred_basis_img = (np.concatenate(pred_basis_img, axis=1).clip(0,1)*255).astype(np.uint8)
                    pred_basis_acc = (np.concatenate(pred_basis_acc, axis=1).clip(0,1)*255).astype(np.uint8)
                    pred_basis_color = (np.concatenate(pred_basis_color, axis=1).clip(0,1)*255).astype(np.uint8)
                    
                    if not eval_time:
                        if write_video:
                            all_preds_basis_img.append(pred_basis_img)
                            all_preds_basis_acc.append(pred_basis_acc)
                            all_preds_basis_color.append(pred_basis_color)
                            all_preds_view_dep_color.append(pred_view_dep_color)
                        else:
                            imageio.imwrite(os.path.join(save_path, f'{name}_{t:04d}_basis_img.png'), pred_basis_img)
                            imageio.imwrite(os.path.join(save_path, f'{name}_{t:04d}_basis_acc.png'), pred_basis_acc)
                            imageio.imwrite(os.path.join(save_path, f'{name}_{t:04d}_basis_color.png'), pred_basis_color)
                            imageio.imwrite(os.path.join(save_path, f'{name}_{t:04d}_view_dep_color.png'), pred_view_dep_color)
                            
                if not eval_time:
                    if write_video:
                        all_preds.append(pred)
                        all_preds_depth.append(pred_depth)
                        imageio.imwrite(os.path.join(save_path,'imgs', f'{name}_{t:04d}_rgb.png'), pred)
                    else:
                        imageio.imwrite(os.path.join(save_path, f'{name}_{t:04d}_rgb.png'), pred)
                        imageio.imwrite(os.path.join(save_path, f'{name}_{t:04d}_depth.png'), pred_depth)

                pbar.update(loader.batch_size)
            # tok = time.time()
            toc.record()
            torch.cuda.synchronize()
            tik = tic.elapsed_time(toc)
            print("\nUsed time: %.4f, average time: %.4f ms"%(tik, tik/len(loader)))
            # print("Used time: %.4f, average time: %.4f"%(tok-tik, (tok-tik)/len(loader)))
        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)

            def mwrite(filename, frames):
                frames = frames[:, :frames.shape[1]//2*2, :frames.shape[2]//2*2]
                imageio.mimwrite(filename, frames, fps=25, quality=8, macro_block_size=1)
            mwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds)
            mwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth)

        self.log(f"==> Finished Test.")
      
    def train_gui(self,cur_epoch,stylizer,train_loader,valid_loader,max_epoch_Photo,max_epoch_nonPhoto, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1, stu_color=False, gui_mode=True):
        #* Forgive me for the long parameter list 
        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale
        total_epochs = max_epoch_Photo + max_epoch_nonPhoto
        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)
        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
        }
        self.model.train()
        if cur_epoch == 0:
            self.model.freeze_teacher()
            self.model.stylizer = stylizer
            self.model.stylizer.parse_style()
            # # print(self.model.stylizer.style_info_ls)
            # exit()
            self.model.stylizer.update_delta_hsv(rgb_orig=self.model.basis_color) 
            if self.opt.recolor:
                self.model.basis_color = self.model.stylizer.recolor(self.model.basis_color)
            self.error_map = train_loader._data.error_map
            
        if cur_epoch == max_epoch_Photo:
            self.reset_lr()
            self.model.stylizer.parse_style()
            self.model.photo_mode = False
            self._init_non_photo_stylize(valid_loader)#,style_info_ls=self.style_info_ls)
            
        if cur_epoch < max_epoch_Photo:
            self.model.photo_mode = True
            # print(self.model.stylizer.style_dirs)
            self.distill_one_epoch(train_loader) 
        else:            
            self.model.photo_mode = False
            self.stylize_one_epoch(valid_loader)
            self.model.photo_mode = True
        
        self.model.eval()

        with torch.no_grad():
            # with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed! (but not perturb the first sample)
            output_dict = self.test_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp, gui_mode=gui_mode, stu_color=stu_color)
        preds = output_dict['preds'].reshape(-1, rH, rW, 3).clamp(0, 1)
        preds_depth = output_dict['preds_depth'].reshape(-1, rH, rW)
        pred_xyz = output_dict['preds_xyz'].reshape(-1, rH, rW, 3)

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)
            pred_xyz = F.interpolate(pred_xyz.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()

        if self.opt.color_space == 'linear':
            preds = linear_to_srgb(preds)

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()
        pred_xyz = pred_xyz[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
            'xyz': pred_xyz,
        }
        return outputs

    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1, stu_color=False, gui_mode=True):
        
        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)
        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
        }
        
        self.model.eval()

        with torch.no_grad():
            # with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed! (but not perturb the first sample)
            output_dict = self.test_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp, gui_mode=gui_mode, stu_color=stu_color)
        preds = output_dict['preds'].reshape(-1, rH, rW, 3).clamp(0, 1)
        preds_depth = output_dict['preds_depth'].reshape(-1, rH, rW)
        pred_xyz = output_dict['preds_xyz'].reshape(-1, rH, rW, 3)

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)
            pred_xyz = F.interpolate(pred_xyz.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()

        if self.opt.color_space == 'linear':
            preds = linear_to_srgb(preds)

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()
        pred_xyz = pred_xyz[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
            'xyz': pred_xyz,
        }

        return outputs
            
    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, file_path)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    # we don't consider continued training from the best ckpt, so we discard the unneeded density_grid to save some storage (especially important for dnerf)
                    if 'density_grid' in state['model']:
                        del state['model']['density_grid']

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']
            else:
                self.log(f"[WARN] mean_density not found, generating new density grid...")   
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
        if self.opt.test:
            self.model.basis_color = self.model.original_color

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
        

    def load_palette_checkpoint(self, ckpt_path):
        checkpoint_list = sorted(glob.glob(f'{ckpt_path}/checkpoints/palette_ep*.pth'))
        assert(checkpoint_list)

        checkpoint = checkpoint_list[-1]
        self.log(f"[INFO] Latest checkpoint is {checkpoint}")

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        # ic(missing_keys, unexpected_keys) # correct
        self.log("[INFO] unexpected_keys:", unexpected_keys)
        #todo: detect how many color palettes are needed
        # assert(unexpected_keys==['basis_color_origin']) # not used, OK to be dummy

        self.log("[INFO] loaded nerf model.")
        if len(missing_keys) > 0:
            self.log(f"Missing keys should be fine.")        
        
        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']
       
    def stylize(self,valid_loader, max_epochs):
        #* before stylize, make sure you run distill(), as it will store occupacy grid information for stylize method
        #* parameters updated in distill(): self.hard_rays_pool, self.error_map
        self.reset_lr()
        self.model.photo_mode = False
        self._init_non_photo_stylize(valid_loader) #* KY: load and resize style images
        
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.stylize_one_epoch(valid_loader)
            
            # if self.scheduler_update_every_step:
            #     self.lr_scheduler.step()

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def stylize_one_epoch(self, loader):
        self.log(f"==> Start Non-Photorealistic Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")
        
        total_loss = 0

        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()
        
        self.model.train() # set self.training = True
        
        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            self.local_step += 1
            self.global_step += 1
            
            #todo loss was backward here
            loss, loss_dict = self.stylize_step(data) 

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), nn_loss={loss_dict['nn_loss']:.4f}, tv_loss={loss_dict['tv_loss']:.4f}, content={0.005*loss_dict['content']:.4f}, lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            
        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")
    
    
    def stylize_step(self, data):
        rays_o = data["rays_o"]  # [B, N, 3]
        rays_d = data["rays_d"]  # [B, N, 3]  [1, N=rays_num=4096, 3]
        # ic(rays_d.shape, rays_o.shape)
        
        images = data["images"]  # [B, N, 3/4] #* we do not use images as GT, just for debuging
        B, H, W, C = images.shape
        loss_dict = {'content':0.0,'style':0.0,'nn_loss':0.0,'tv_loss':0.0}
        loss = 0.0
        # import pdb; pdb.set_trace()
        style_bg_color = [self.style_info_ls[i]['bg_color'] for i in range(self.opt.num_basis)]
        def compute_image_loss():
            #* compute_image_loss here (style loss)
            with torch.no_grad():
                basis_weights_sum,basis_masks,basis_images,basis_contents = self.model.run_cuda_seg_image(rays_o, rays_d, dt_gamma=0, perturb=False, max_steps=1024, T_thresh=1e-4, bg_color=style_bg_color)
                for i in range(self.opt.num_basis): # reshaoe to [B, H, W, 3]
                    # basis_images[i] = basis_images[i].view(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
                    basis_images[i] = basis_images[i].view(H, W, 3).permute(2, 0, 1).unsqueeze(0).contiguous() # 3 H W
                    basis_contents[i] = basis_contents[i].view(H, W, 3).permute(2, 0, 1).unsqueeze(0).contiguous()
                    # basis_masks[i] = basis_masks[i].view(B, H, W)
                    basis_masks[i] = basis_masks[i].view(H, W).unsqueeze(0).contiguous()
                    basis_weights_sum[i] = basis_weights_sum[i].view(B, H, W)
                    
                # if self.global_step%(10)==0: #todo: for debug use
                #     _,_,basis_styles,_ = self.model.run_cuda_seg_image(rays_o, rays_d, dt_gamma=0, perturb=False, max_steps=1024, T_thresh=1e-4, bg_color=[0.95,0.9])
                #     for i in range(self.opt.num_basis):
                #         import PIL
                #         basis_styles[i] = basis_styles[i].view(H, W, 3).permute(2, 0, 1).unsqueeze(0).contiguous() # 3 H W
                #         basis_img = basis_styles[i].squeeze().permute(1, 2, 0).clamp(0, 1).contiguous().clone().detach().cpu().numpy()
                #         img = PIL.Image.fromarray((basis_img*255).astype(np.uint8))
                #         img.save(f'basis_img_{i}.png')
                    
            basis_rgb_grad = []
            for i in range(self.opt.num_basis):
                rgb_pred = basis_images[i]
                rgb_content = basis_contents[i]
                rgb_pred.requires_grad_(True)
                style_img = self.style_info_ls[i]['masked_img']
                
                #* uncomment these lines to enable TV loss
                w_variance = torch.mean(torch.pow(rgb_pred[:, :, :, :-1] - rgb_pred[:, :, :, 1:], 2))
                h_variance = torch.mean(torch.pow(rgb_pred[:, :, :-1, :] - rgb_pred[:, :, 1:, :], 2))
                img_tv_loss = 1.0 * (h_variance + w_variance) / 2.0
                
                nn_loss, _, content_loss = self.nn_loss_fn( 
                    F.interpolate(
                        rgb_pred,
                        size=None,
                        scale_factor=0.5, #was 0.5, this will control the color intensity
                        mode="bilinear",
                    ),
                    style_img.permute(2, 0, 1).unsqueeze(0),
                    loss_names=["nn_loss","content_loss"],
                    contents=F.interpolate(
                        rgb_content,
                        size=None,
                        scale_factor=0.5, #was 0.5, this will control the color intensity
                        mode="bilinear",
                    )
                )
                
                loss = nn_loss+ img_tv_loss + 0.000*content_loss
        
                loss.backward(retain_graph=True)
                loss_dict['style'] += loss.item()
                loss_dict['nn_loss'] += nn_loss.item()
                loss_dict['content'] += content_loss.item()*0.000 #* optional, but we don't use content loss in exp
                loss_dict['tv_loss'] += img_tv_loss.item()
                # ic(loss) #pdvgg: 1.735  #vgg: 1.5002
                # ic(rgb_pred.shape)
                rgb_pred_grad = (self.style_weight[i]*rgb_pred.grad).squeeze(0).permute(1, 2, 0).contiguous().clone().detach().view(-1, 3)
                # rgb_pred_grad = rgb_pred.grad.squeeze(0).permute(1, 2, 0).contiguous().clone().detach().view(-1, 3)
                # rgb_pred = rgb_pred.squeeze(0).permute(1, 2, 0).contiguous().clone().detach()
                # ic(rgb_mask.shape, rgb_pred_grad.shape, rgb_pred.shape)
                # exit()
                basis_rgb_grad.append(rgb_pred_grad)
            return loss, loss_dict, basis_rgb_grad
        # ic(basis_rgb_grad[0].shape)
        # for i in range(self.opt.num_basis):
        #     grad = basis_rgb_grad[i].reshape(800,800,3)
        #     plt.figure()
        #     plt.imshow(grad.cpu().numpy())
        #     plt.show()
        # exit()
        # plt.figure()
        #* backwards and content loss here
        self.optimizer.zero_grad()
        
        loss,loss_dict,basis_rgb_grad = compute_image_loss()

        
        for batch_start in range(0, H*W, self.opt.batch_size):
            # ic(rays_d.shape, rays_o.shape, batch_start, batch_start+self.opt.batch_size)
            outputs = self.model.render(rays_o[:,batch_start:batch_start+self.opt.batch_size],
                                            rays_d[:,batch_start:batch_start+self.opt.batch_size],
                                            staged=False, bg_color=1, perturb=False, force_all_rays=False, **vars(self.opt)) 
            for i in range(self.opt.num_basis):
                rgb_pred = outputs[i]['pred_rgb']
                content_loss = (1 - self.style_weight[i])*self.criterion(outputs[i]['tea_color'],outputs[i]['stu_color']).mean()
                rgb_pred.backward(basis_rgb_grad[i][batch_start:batch_start+self.opt.batch_size],retain_graph=True)
                content_loss.backward(retain_graph=True)
        self.optimizer.step()

        return loss, loss_dict
        
    def _init_non_photo_stylize(self,loader):
        self.epoch = 0
        # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: max(0.1 ** min(epoch / self.opt.style_epoch, 1), self.opt.minlr/self.opt.lr))
        # self.lr_scheduler = scheduler(self.optimizer)
        self.model.freeze_teacher() #* just ensure teacher model is frozen
        # for name, param in self.model.named_parameters(): #correct
        #     if param.requires_grad:
        #         print(name)
        # exit()
        self.nn_loss_fn = CalculateLoss(device=self.device)
        self.style_info_ls = self.model.stylizer.styles
        contentH, contentW = loader._data.H, loader._data.W
        content_long_side = max([contentH, contentW])
        #* resize style image such that its long side matches the long side of content images
        for style_idx, style in enumerate(self.style_info_ls):
            style_mask = style['mask']
            style_masked_img = style['masked_img']
            # style_masked_img = style['img']
            style_h, style_w = style_masked_img.shape[:2]
            if style_h > style_w: 
                style_masked_img = cv2.resize(
                    style_masked_img,
                    (int(content_long_side / style_h * style_w), content_long_side),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                style_masked_img = cv2.resize(
                    style_masked_img,
                    (content_long_side, int(content_long_side / style_w * style_h)),
                    interpolation=cv2.INTER_AREA,
                )
            style_masked_img = cv2.resize( # seems trival for adjusting
                style_masked_img,
                (style_masked_img.shape[1] // 2, style_masked_img.shape[0] // 2), #* was //2
                interpolation=cv2.INTER_AREA, #* INTER_AREA by default
            )  # this is to replace the downsampling in optimization loop
            savePath = os.path.join(self.workspace, f"style_img_{style_idx}.png")
            
            preview_img = cv2.cvtColor(style_masked_img.astype(np.float32), cv2.COLOR_BGR2RGB)
            cv2.imwrite(savePath,preview_img)
            
            style_masked_img = (style_masked_img / 255.0).astype(np.float32)
            self.style_info_ls[style_idx]['masked_img'] = torch.from_numpy(style_masked_img).to(device=self.device)
            self.style_info_ls[style_idx]['mask'] = torch.from_numpy(style_mask).to(device=self.device)
     
            
            
def preview_img(img):
    import matplotlib.pyplot as plt
    img = img.astype(np.float32)
    plt.imshow(img)
    plt.show()