# import kornia
import torch
from torchvision import models, transforms
import torch.nn.functional as F
from icecream import ic
import numpy as np
from .pdvgg import pdvgg16
from einops import rearrange, repeat
import matplotlib.pyplot as plt
class VGG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.pdvgg = pdvgg16(pretrained=True).eval()
        self.pdvgg = models.vgg16(weights="DEFAULT").eval() #* uncomment this line to use standard convolution
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def get_hypercolumn_feat(self,x,layers,scale,supress_assert=True):
        #todo: implement input rotation 
        imgH, imgW = x.shape[2], x.shape[3]
        if not supress_assert:
            assert x.min() >= 0.0 and x.max() <= 1.0, "input is expected to be an image scaled between 0 and 1"
        x = self.normalize(x)
        final_ix = max(layers)
        outputs = []
        for ix, layer in enumerate(self.pdvgg.features):
            # normalize with # of channels and resize its spatial size
            x = layer(x)
            if ix in layers: 
                outputs.append(F.interpolate(x/x.shape[1], size=(int(imgH*scale), int(imgW*scale)), mode='bilinear', align_corners=False))
            if ix == final_ix:
                break
        return outputs

    def get_feats(self, x, layers=[], supress_assert=True, mask_in=None):
        # Layer indexes:
        # Conv1_*: 1,3
        # Conv2_*: 6,8
        # Conv3_*: 11, 13, 15
        # Conv4_*: 18, 20, 22
        # Conv5_*: 25, 27, 29

        if not supress_assert:
            assert x.min() >= 0.0 and x.max() <= 1.0, "input is expected to be an image scaled between 0 and 1"
        mask = mask_in
        x = self.normalize(x)
        final_ix = max(layers)
        outputs = []
        outputs_masks = []
        for ix, layer in enumerate(self.pdvgg.features):
            x = layer(x)
            if ix in layers:
                outputs.append(x)
                outputs_masks.append(mask)

            if ix == final_ix:
                break
        # exit()
        return outputs

def cos_distance(a, b, center=True):
    """a: [b, c, hw],
    b: [b, c, h2w2]
    """
    # """cosine distance
    if center:
        a = a - a.mean(2, keepdims=True)
        b = b - b.mean(2, keepdims=True)

    a_norm = ((a * a).sum(1, keepdims=True) + 1e-8).sqrt()
    b_norm = ((b * b).sum(1, keepdims=True) + 1e-8).sqrt()

    a = a / (a_norm + 1e-8)
    b = b / (b_norm + 1e-8)

    d_mat = 1.0 - torch.matmul(a.transpose(2, 1), b)

    """
    a_norm_sq = (a * a).sum(1).unsqueeze(2)
    b_norm_sq = (b * b).sum(1).unsqueeze(1)

    d_mat = a_norm_sq + b_norm_sq - 2.0 * torch.matmul(a.transpose(2, 1), b)
    """
    return d_mat



def cos_loss(a, b):
    # """cosine loss
    a_norm = (a * a).sum(1, keepdims=True).sqrt()
    b_norm = (b * b).sum(1, keepdims=True).sqrt()
    a_tmp = a / (a_norm + 1e-8)
    b_tmp = b / (b_norm + 1e-8)
    cossim = (a_tmp * b_tmp).sum(1)
    cos_d = 1.0 - cossim
    return cos_d.mean()



    


def feat_replace(a, b): # b is style image, a is radiance output
    n, c, h, w = a.size()
    n2, c, h2, w2 = b.size()
    
    assert (n == 1) and (n2 == 1) # n must be 1

    a_flat = a.view(n, c, -1)
    b_flat = b.view(n2, c, -1)
    b_ref = b_flat.clone()

    z_new = []

    # Loop is slow but distance matrix requires a lot of memory
    for i in range(n):
        z_dist = cos_distance(a_flat[i : i + 1], b_flat[i : i + 1])

        z_best = torch.argmin(z_dist, 2)
        del z_dist

        z_best = z_best.unsqueeze(1).repeat(1, c, 1)
        feat = torch.gather(b_ref, 2, z_best)

        z_new.append(feat)

    z_new = torch.cat(z_new, 0)
    z_new = z_new.view(n, c, h, w)
    return z_new


def gram_matrix(feature_maps, center=False):
    """
    feature_maps: b, c, h, w
    gram_matrix: b, c, c
    """
    b, c, h, w = feature_maps.size()
    features = feature_maps.view(b, c, h * w)
    if center:
        features = features - features.mean(dim=-1, keepdims=True)
    G = torch.bmm(features, torch.transpose(features, 1, 2))
    return G


class CalculateLoss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vgg = VGG().to(device)
    
    def forward(
        self,
        outputs,
        styles,
        blocks=[
             2
        ],
        loss_names=["nn_loss"],  # can also include 'gram_loss', 'content_loss', 'moment_loss'
        contents=None,
    ): #momment loss to resolve the saturation problem
        # ic(outputs.shape, styles.shape) #[1,3,400,400] [1,3,400,314]
        blocks.sort()
        block_indexes = [[1, 3], [6, 8], [11, 13, 15], [18, 20, 22], [25, 27, 29]]
        all_layers = []
        for block in blocks:
            all_layers += block_indexes[block]

        x_feats_all = self.vgg.get_feats(outputs, all_layers)
        
        with torch.no_grad():
            s_feats_all = self.vgg.get_feats(styles, all_layers)
            if "content_loss" in loss_names:
                content_feats_all = self.vgg.get_feats(contents, all_layers)
       
        ix_map = {}
        for a, b in enumerate(all_layers):
            ix_map[b] = a
        nn_loss = 0.0
        gram_loss = 0.0
        magnitude_loss = 0.0
        content_loss = 0.0
     
        for block in blocks:
            layers = block_indexes[block]
            x_feats = torch.cat([x_feats_all[ix_map[ix]] for ix in layers], 1)
            s_feats = torch.cat([s_feats_all[ix_map[ix]] for ix in layers], 1)
            
            # x_masks = torch.cat([x_feats_masks[ix_map[ix]] for ix in layers], 1)
            # s_masks = torch.cat([s_feats_masks[ix_map[ix]] for ix in layers], 1)
            # x_masks = self.downsample(outputs_masks, x_feats.shape[-2:])
            # s_masks = self.downsample(styles_masks, s_feats.shape[-2:])
            
            if "nn_loss" in loss_names:
                # nn_loss += matching_nn_loss(x_feats, s_feats, x_masks, s_masks)
                target_feats = feat_replace(x_feats, s_feats)
                nn_loss += cos_loss(x_feats, target_feats)
            if "gram_loss" in loss_names:
                gram_loss += torch.mean((gram_matrix(x_feats) - gram_matrix(s_feats)) ** 2)
            if "content_loss" in loss_names:
                content_feats = torch.cat([content_feats_all[ix_map[ix]] for ix in layers], 1)
                content_loss += torch.mean((content_feats - x_feats) ** 2)


        return nn_loss, gram_loss,content_loss
    
    def downsample(self, x, new_dim):
        H, W = x.shape
        NH, NW = new_dim
        r_indices = torch.linspace(0, H-1, NH).long()
        c_indices = torch.linspace(0, W-1, NW).long()
        return x[r_indices[:, None], c_indices]
    
def cosine_dists(feats1, feats2):
    # ic(feats1.shape, feats2.shape)
    feats1 = feats1.squeeze()
    feats2 = feats2.squeeze()
    # feats1 = torch.sum(feats1, dim=0)
    # feats2 = torch.sum(feats2, dim=0)
    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(feats1.cpu().detach().numpy())
    # axs[1].imshow(feats2.cpu().detach().numpy())
    # plt.show()
    # exit()
    
    feats1 = rearrange(feats1.squeeze(), 'c h w -> (h w) c')
    feats2 = rearrange(feats2.squeeze(), 'c h w -> (h w) c')
    
    feats1_hat = feats1 / (torch.linalg.norm(feats1, dim=1) )[:, None]
    feats2_hat = feats2 / (torch.linalg.norm(feats2, dim=1) )[:, None]
    dists = 1.0 - torch.matmul(feats1_hat, feats2_hat.T)
    # feats1_norm = (feats1 * feats1).sum(1, keepdims=True).sqrt()
    # feats2_norm = (feats2 * feats2).sum(1, keepdims=True).sqrt()
    # a = feats1 / (feats1_norm + 1e-8)
    # b = feats2 / (feats2_norm + 1e-8)
    # dists = 1.0 - torch.matmul(a, b.T)
    return dists
        
def matching_nn_loss(x_feat, s_feat, x_mask, s_mask):
    dists  = cosine_dists(x_feat, s_feat)
    x_mask = (x_mask == 1).reshape(-1)
    s_mask = (s_mask == 1).reshape(-1)
    invalid_mask = torch.logical_and(*torch.meshgrid(x_mask, s_mask))
    dists[invalid_mask] = float('inf')
    # ic(dists)
    min_dists = torch.amin(dists, dim=1)
    # ic(min_dists)
    loss = torch.mean(min_dists)
    return loss