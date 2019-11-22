import os, sys, math, random, itertools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.checkpoint import checkpoint


class GeoNetNormalToDepth(nn.Module):
    '''
        This model implements DEPTH -> NORMAL consistency for GeoNet. 
        It has no learnable parameters.
    '''
    def __init__(self, depth_refinement_alpha=0.95, depth_refinement_beta=9):
        super().__init__()
        self.depth_refinement_network = KernelRegression(alpha=depth_refinement_alpha, beta=depth_refinement_beta)


    def forward(self, x, eps=1e-5):
        initial_depth, x_fov, initial_normals = x
        initial_normals = initial_normals * 2 - 1.0
        initial_depth[initial_depth < eps] = eps # Avoid divide-by-zero errors in reprojection 

        refined_depth = self.depth_refinement_network(-initial_normals, initial_depth, x_fov) # note our convention is inverted
        refined_depth = refined_depth.clamp(min=0.0, max=1.0)
        return refined_depth


class GeoNetDepthToNormal(nn.Module):
    '''
        This model implements NORMAL -> DEPTH consistency for GeoNet.
        This has learnable parameters (in the residual module) that require pretraining.         
    '''
    def __init__(self, normal_refinement_gamma=0.95, normal_refinement_beta=9):
        '''
            
        '''
        super().__init__()
        self.normal_refinement_network_1 = LeastSquareModule(gamma=normal_refinement_gamma, beta=normal_refinement_beta)
        self.normal_refinement_network_2 = NormalResidual()
   
    def forward(self, x, eps=1e-5):
        initial_depth, x_fov, initial_normals = x
        initial_depth[initial_depth < eps] = eps # Avoid divide-by-zero errors in reprojection 
        with torch.no_grad():
            normal_from_depth = self.normal_refinement_network_1(initial_depth, x_fov)
        refined_normals = self.normal_refinement_network_2(initial_normals, normal_from_depth)
        refined_normals = (refined_normals + 1.0) / 2.0
        refined_normals = refined_normals.clamp(min=0.0, max=1.0)

        return refined_normals

    
    
    
       
def reproject_depth(depth, field_of_view, cached_cr=None, max_depth=1.):
    """Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    depth is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.

    """


    dx, dy = torch.tensor(depth.shape[2:4]) - 1
    cx, cy = torch.tensor([dx, dy]) / 2

    fx, fy = torch.tensor([[depth.shape[2]], [depth.shape[3]]], device=field_of_view.device, dtype=torch.float32) \
                / (2. * torch.tan(field_of_view.float() / 2.).unsqueeze(0))

    if cached_cr is None:
        cols, rows = depth.shape[2], depth.shape[3]
        c, r = torch.tensor(np.meshgrid(np.arange(cols), np.arange(rows), sparse=False), device=field_of_view.device, dtype=torch.float32)
    else:
        c, r = cached_cr

    z = depth.squeeze(1) * max_depth
    x = z * ((c - cx).unsqueeze(0) / fx.unsqueeze(1).unsqueeze(1))
    y = z * ((r - cy).unsqueeze(0) / fy.unsqueeze(1).unsqueeze(1))
    return torch.stack((x, y, z), dim=1), cached_cr


    
def kernel_regress(x_normal, x_depth3d, size=1, alpha=0.95):
    '''
        Regress depth using kernel weights from normals
        x_normal: [batch_size, 3, width, height]
        x_depth3d: xyz coords [batch_size, 3, width, height]
    ''' 
    # Expand patches
    stride=1

    x_normal = x_normal / torch.norm(x_normal, dim=1, keepdim=True)
    normal_padded = F.pad(x_normal, (size//2, size//2, size//2, size//2), mode='replicate')
    normal_patches = normal_padded.unfold(2, size, stride).unfold(3, size, stride) # [batch_size, 3, width, height, size, size]
    normal_patches = normal_patches.reshape((*normal_patches.shape[:-2], ) + (-1,))  # [batch_size, 3, width, height, size*size]
    normals = x_normal.unsqueeze(-1) 

    # Get depth patches
    depth_padded = F.pad(x_depth3d, (size//2, size//2, size//2, size//2), mode='replicate')
    depth_patches = depth_padded.unfold(2, size, stride).unfold(3, size, stride)  # [batch_size, 1, width, height, size, size]
    depth_patches = depth_patches.reshape((*depth_patches.shape[:-2], ) + (-1,))  # [batch_size, 1, width, height, size*size]

    # Calculate kernel weights
    weight_kernelized = torch.sum(normals * normal_patches, dim=1)  
#     print('\tkernel weights:', weight_kernelized.min(), weight_kernelized.max())
    
    # Set out-of-plane points to have zero weight
    weight_kernelized[weight_kernelized < alpha] = 0.0
    weight_kernelized = weight_kernelized.unsqueeze(1)

    # Use weights to regress depth
#     x_depth3d[x_depth3d < eps] = eps
    depth_votes_denom = torch.sum(normals * (x_depth3d / x_depth3d[:, 2].unsqueeze(1)).unsqueeze(-1), dim=1)
    depth_votes = torch.sum(normals * depth_patches, dim=1) / depth_votes_denom

    depth_pred = torch.sum(depth_votes.unsqueeze(1) * weight_kernelized, dim=-1)
    sums = torch.sum(weight_kernelized, dim=-1)
    depth_pred = depth_pred / sums
    
    depth_pred[depth_pred != depth_pred] = 1e-3
#     print('\tdepth_pred', depth_pred.min(), depth_pred.max())
    return depth_pred

def lstq(A, Y, lamb=0.01):
        """
        Differentiable least square
        :param A: m x n
        :param Y: n x 1
        """
        # Assuming A to be full column rank
        cols = A.shape[1]
        print (torch.matrix_rank(A))
        if cols == torch.matrix_rank(A):
            q, r = torch.qr(A)
            x = torch.inverse(r) @ q.T @ Y
        else:
            A_dash = A.permute(1, 0) @ A + lamb * torch.eye(cols)
            Y_dash = A.permute(1, 0) @ Y
            x = lstq(A_dash, Y_dash)
        return x
    
def least_square_normal_regress(x_depth3d, size=9, gamma=0.15, depth_scaling_factor=1, eps=1e-5):    
    stride=1

    # xyz_perm = xyz.permute([0, 2, 3, 1])
    xyz_padded = F.pad(x_depth3d, (size//2, size//2, size//2, size//2), mode='replicate')
    xyz_patches = xyz_padded.unfold(2, size, stride).unfold(3, size, stride) # [batch_size, 3, width, height, size, size]
    xyz_patches = xyz_patches.reshape((*xyz_patches.shape[:-2], ) + (-1,))  # [batch_size, 3, width, height, size*size]
    xyz_perm = xyz_patches.permute([0, 2, 3, 4, 1])

    diffs = xyz_perm - xyz_perm[:, :, :, (size*size)//2].unsqueeze(3)
    diffs = diffs / xyz_perm[:, :, :, (size*size)//2].unsqueeze(3)
    diffs[..., 0] = diffs[..., 2]
    diffs[..., 1] = diffs[..., 2]
    xyz_perm[torch.abs(diffs) > gamma] = 0.0

    A_valid = xyz_perm * depth_scaling_factor                           # [batch_size, width, height, size, 3]

    # Manual pseudoinverse
    A_trans = xyz_perm.permute([0, 1, 2, 4, 3]) * depth_scaling_factor  # [batch_size, width, height, 3, size]
    A = torch.matmul(A_trans, A_valid)

    #try:
    #    A_det = torch.det(A)
    #    A[A_det < eps, :, :] = torch.eye(3)
    #except:
    #    pass

    #A_inv = torch.inverse(A)
    #b = torch.ones(list(A_valid.shape[:4]) + [1])
    #lstsq = A_inv.matmul(A_trans).matmul(b)
    #lstsq = lstsq.to(x_depth3d.device)

    #b = torch.ones(list(A_valid.shape[:4]) + [1], device='cpu')
    #lstsq = torch.pinverse(A_valid.cpu()).matmul(b)
    #lstsq = lstsq.to(x_depth3d.device)


    try:
        A_det = torch.det(A)
        A[A_det < eps, :, :] = torch.eye(3).to(x_depth3d.device)
    except:
        pass
    
    try: # Fast, but MAGMA sometimes complains 
        A_inv = torch.inverse(A)
        b = torch.ones(list(A_valid.shape[:4]) + [1], device=x_depth3d.device)
        lstsq = A_inv.matmul(A_trans).matmul(b)
    except:
        # More stable, but slow (and must be done on cpu)
        return torch.zeros((x_depth3d.shape[0], 3, 256, 256)).to(x_depth3d.device) # give up
        for i in range(100):
            try:
                b = torch.ones(list(A_valid.shape[:4]) + [1], device='cpu')
                lstsq = torch.pinverse(A_valid.cpu()).matmul(b).to(x_depth3d.device)
                break
            except:
                pass
    
    #try:
    #    A_det = torch.det(A)
    #    A[A_det < eps, :, :] = torch.eye(3).to(x_depth3d.device)
    #except:
    #    pass

    #try: # Fast, but MAGMA sometimes complains 

    #    A_inv = torch.inverse(A)
    #    b = torch.ones(list(A_valid.shape[:4]) + [1], device=x_depth3d.device)
    #    lstsq = A_inv.matmul(A_trans).matmul(b)
    #except:
    #    # More stable, but slow (and must be done on cpu)
    #    for i in range(100):
    #        try:
    #            b = torch.ones(list(A_valid.shape[:4]) + [1], device='cpu')
    #            lstsq = torch.pinverse(A_valid.cpu()).matmul(b).to(x_depth3d.device)
    #            break
    #        except:
    #            pass






    # Alternatively, one could use the following functions--but these are slow on GPU.
#     b = torch.ones(list(A_valid.shape[:4]) + [1], device=x_depth3d.device)
#     lstsq = torch.solve(b, A_valid)
#     q, r = torch.qr(A_valid)
#     lstsq = torch.inverse(r) @ q.permute([0, 1, 2, 4, 3]) @ b
#     lstsq = torch.pinverse(A_valid.cpu()).to(b.device).matmul(b)


#     print("---")
#     print('A_valid', A_valid.shape)
#     print('A shape:', A.shape)
#     print('A_inv', A_inv.shape)
#     print('A_trans', A_trans.shape)
#     print('out', lstsq.shape)
#     print('norm', torch.norm(lstsq, dim=3).unsqueeze(3).shape)

    lstsq = lstsq / torch.norm(lstsq, dim=3).unsqueeze(3)
    lstsq[lstsq != lstsq] = 0.0
    return -lstsq.squeeze(-1).permute([0, 3, 1, 2])
    
    

class KernelRegression(nn.Module):

    def __init__(self, alpha=0.95, beta=9):
        self.cached_cr = None
        self.shape = None
        self.patch_size = beta
        self.in_plane_thresh = alpha
        super().__init__()

    
    def forward(self, x_normal, x_depth, field_of_view_rads):
        im_shape = x_normal.shape[2], x_normal.shape[3]
#         if self.cached_cr is None or self.shape != im_shape:
#             cols, rows = im_shape
#             self.cached_cr = torch.tensor(np.meshgrid(np.arange(cols), np.arange(rows), sparse=False), device=field_of_view_rads.device, dtype=torch.float32)
#             self.shape = im_shape
        
        x_depth3d, cached_cr = reproject_depth(x_depth, field_of_view_rads, cached_cr=self.cached_cr, max_depth=1.)
        if self.cached_cr is None:
            self.cached_cr = cached_cr
        return kernel_regress(x_normal, x_depth3d, self.patch_size, self.in_plane_thresh, )


class LeastSquareModule(nn.Module):

    def __init__(self, gamma=0.15, beta=9):
        self.cached_cr = None
        self.shape = None
        self.patch_size = beta
        self.z_depth_thresh = gamma
        super().__init__()

    
    def forward(self, x_depth, field_of_view_rads):
#         im_shape = x_depth.shape[2], x_depth.shape[3]
#         if self.cached_cr is None or self.shape != im_shape:
#             cols, rows = im_shape
#             self.cached_cr = torch.tensor(np.meshgrid(np.arange(cols), np.arange(rows), sparse=False), device=field_of_view_rads.device)
#             self.shape = im_shape
        x_depth3d, cached_cr = reproject_depth(x_depth, field_of_view_rads, cached_cr=self.cached_cr, max_depth=1.)
        if self.cached_cr is None:
            self.cached_cr = cached_cr
        return least_square_normal_regress(x_depth3d, size=self.patch_size, gamma=self.z_depth_thresh)



class NormalResidual(nn.Module):
    
    def __init__(self, in_size=(256, 256)):
        super().__init__()
#         self.normal_from_depth = LeastSquareModule(gamma, beta)

        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.gn1_1 = nn.GroupNorm(8, 64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.gn1_2 = nn.GroupNorm(8, 64)        
        self.conv1_3 = nn.Conv2d(64, 3, 3, padding=1)
        
        self.fc1 = nn.Conv2d(6, 3, 1, padding=0)


#         self.max_pool = nn.MaxPool2d(3, 3)
#         self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
#         self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

#         self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
#         self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
#         self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)


#         self.upsample = nn.Upsample(size=in_size)
        

#         self.conv4_1 = nn.Conv2d(128, 128, 3, padding=1, dilation=2)
#         self.conv4_2 = nn.Conv2d(128, 128, 3, padding=1, dilation=2)
#         self.conv4_3 = nn.Conv2d(128, 128, 3, padding=1, dilation=2)
#         self.conv4_4 = nn.Conv2d(128, 128, 3, padding=1)
#         self.conv4_5 = nn.Conv2d(128, 128, 3, padding=1)
#         self.conv4_6 = nn.Conv2d(128, 128, 3, padding=1)
#         self.conv4_7 = nn.Conv2d(128, 3, 3, padding=1)



    def forward(self, x_normal, normal_from_depth):
        
        # Residual
        normal_from_depth_scaled = normal_from_depth * 10 # GeoNet orig paper implements this way

        res = F.relu(self.conv1_1(normal_from_depth_scaled))
        res = self.gn1_1(res)
        res = F.relu(self.conv1_2(res))
        res = self.gn1_2(res)
        res = self.conv1_3(res)

#         res = self.max_pool(res)
#         res = self.conv2_1(res)
#         res = self.conv2_2(res)
#         res = self.conv3_1(res)
#         res = self.conv3_2(res)
#         res = self.conv3_3(res)
#         res = self.fc1(res)
#         res = self.upsample(res)
        
        # Sum

        sum_norm_noise = normal_from_depth + res
        normed = torch.norm(sum_norm_noise, dim=1, keepdim = True)
#         print('analytic, residual, normed min', normal_from_depth.min().item(), res.min().item(), normed.min().item())
        sum_norm_noise = sum_norm_noise / normed

        # Concat and FC
        norm_pred_all = torch.cat([x_normal, sum_norm_noise], dim=1)
        res = self.fc1(norm_pred_all)

        
        return res / torch.norm(res, dim=1, keepdim=True)


    
class GeoNet(nn.Module):
    
    def __init__(self, depth_prediction_network_fn, depth_prediction_network_fn_kwargs,
                       normal_prediction_network_fn, normal_prediction_network_fn_kwargs,
                       depth_refinement_alpha=0.95, depth_refinement_beta=9,
                       normal_refinement_gamma=0.95, normal_refinement_beta=9,
                    ):
        '''
            
        '''
        super().__init__()
        self.nets = nn.ModuleList()
        self.depth_network = depth_prediction_network_fn(**depth_prediction_network_fn_kwargs)
        self.depth_refinement_network = KernelRegression(alpha=depth_refinement_alpha, beta=depth_refinement_beta)
        self.normal_network = normal_prediction_network_fn(**normal_prediction_network_fn_kwargs)

        self.normal_refinement_network_1 = LeastSquareModule(gamma=normal_refinement_gamma, beta=normal_refinement_beta)
        self.normal_refinement_network_2 = NormalResidual()


    def initialize_from_checkpoints(self, checkpoint_paths, logger=None):
        for i, (net, ckpt_fpath) in enumerate(zip(self.nets, checkpoint_paths)):
            if logger is not None:
                logger.info(f"Loading step {i} from {ckpt_fpath}")
            checkpoint = torch.load(ckpt_fpath)
            sd = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
            net.load_state_dict(sd)        
        return self
    
    def forward(self, x_rgb, x_fov, eps=1e-5):
        initial_depth = self.depth_network(x_rgb)
        initial_normals = self.normal_network(x_rgb)

        
        initial_depth[initial_depth < eps] = eps # Avoid divide-by-zero errors in reprojection 
        refined_depth = self.depth_refinement_network(-initial_normals, initial_depth, x_fov) # note our convention is inverted
        refined_depth = refined_depth.clamp(min=0.0, max=1.0)

        normal_from_depth = self.normal_refinement_network_1(initial_depth, x_fov)
        refined_normals = self.normal_refinement_network_2(initial_normals, normal_from_depth)
        refined_normals = refined_normals.clamp(min=0.0, max=1.0)

        return initial_normals, initial_depth, refined_normals, refined_depth
    

