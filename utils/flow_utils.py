import math
from functools import lru_cache

import cv2
import numpy as np
import torch
import math
import torch.nn.functional as F

from scene_PIDG.gaussian_model import GaussianModel
from scene_PIDG.cameras import Camera

from utils.loss_utils import calculate_flow_loss
from utils.flow_vis_utils import save_vis_flow_tofile

from PIL import Image
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image

def unproject(
    depth_map: torch.Tensor,       # (1, H, W)
    w2c: torch.Tensor,             # (4, 4) World→Cam
    K: torch.Tensor,               # (3, 3)
):
    B, H, W = depth_map.shape      # B==1
    device, dtype = depth_map.device, depth_map.dtype

    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij"
    )                               # (H, W)
    ones = torch.ones_like(u)

    uv1 = torch.stack((u, v, ones), dim=0)          # (3, H, W)
    uv1 = uv1.reshape(3, -1)                        # (3, H·W)

    invK = torch.inverse(K)
    rays_cam = invK @ uv1                           # (3, H·W)
    depth_flat = depth_map.reshape(1, -1)
    xyz_cam  = rays_cam * depth_flat                # (3, H·W)

    c2w = torch.inverse(w2c)
    xyz1_cam = torch.cat([xyz_cam, torch.ones_like(depth_flat)], dim=0)  # (4, H·W)
    xyz1_w   = c2w @ xyz1_cam                       # (4, H·W)
    xyz_w    = xyz1_w[:3].reshape(3, H, W)          # (3, H, W)

    uv_present = torch.stack((u, v), dim=0)         # (2, H, W)

    return uv_present, xyz_w

def project(xyz_world: torch.Tensor, extrinsic_matrix: torch.Tensor, intrinsic_matrix: torch.Tensor):
    """
    Args:
        xyz_world: (3, N)
        extrinsic_matrix: (4, 4) world-to-camera
        intrinsic_matrix: (3, 3) camera-to-pixel
    Returns:
        [(2, N), (1, N)]
    """
    if not isinstance(extrinsic_matrix, torch.Tensor):
        extrinsic_matrix = torch.tensor(extrinsic_matrix, dtype=xyz_world.dtype, device=xyz_world.device)
    if not isinstance(intrinsic_matrix, torch.Tensor):
        intrinsic_matrix = torch.tensor(intrinsic_matrix, dtype=xyz_world.dtype, device=xyz_world.device)
    xyz1_world = torch.cat([xyz_world, torch.ones_like(xyz_world[:1])], dim=0)
    xyz1_cam = extrinsic_matrix @ xyz1_world
    uvz = intrinsic_matrix @ xyz1_cam[:3, :]
    z = uvz[2:]
    uv = uvz[:2] / z
    return uv, z

def readFlow_flo(fn: str):
    """Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            raise ValueError("Magic number incorrect. Invalid .flo file")
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape testdata into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def readFlow(fn: str):
    if fn.endswith(".flo"):
        return readFlow_flo(fn)
    else:
        try:
            return np.load(fn)
        except:
            print(f"Error reading {fn}")
            raise

def warp_field(field_t1,   # (H,W,2) @ t+1
               flow_fwd,   # (H,W,2) @ t
               align_corners=True):
    """
    Warp the vector field defined on the t+1 grid back to the t grid.
    Returns a (H, W, 2) field aligned with frame t.
    """
    H, W, _ = flow_fwd.shape
    device  = flow_fwd.device
    dtype   = flow_fwd.dtype

    # compute the destination coordinates for each pixel in frame t: p4 = p1 + flow_fwd
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )
    u_t1 = u + flow_fwd[..., 0]
    v_t1 = v + flow_fwd[..., 1]

    # normalize coordinates to [-1, 1] for grid_sample
    if align_corners:
        u_norm = 2.0 * u_t1 / (W - 1) - 1.0
        v_norm = 2.0 * v_t1 / (H - 1) - 1.0
    else:
        u_norm = 2.0 * (u_t1 + 0.5) / W - 1.0
        v_norm = 2.0 * (v_t1 + 0.5) / H - 1.0

    grid = torch.stack((u_norm, v_norm), dim=-1).unsqueeze(0)  # (1,H,W,2)

    # sample field_t1 back to frame t using grid_sample
    field_t1 = field_t1.permute(2,0,1).unsqueeze(0)            # (1,2,H,W)
    field_t  = F.grid_sample(field_t1, grid,
                             mode='bilinear',
                             padding_mode='border',
                             align_corners=align_corners)[0]   # (2,H,W)
    return field_t.permute(1,2,0)                              # (H,W,2)


def motion_flow_velocity(iteration, gaussians, viewpoint_cam, render_pkg_re, render_pkg_re_2, render_pkg_re_velocity, save_image_dir, dataset_type):
    flow_loss = torch.tensor(0.0)
    
    flow_fwd, _, occ, occ_bwd = viewpoint_cam.flow
    flow_fwd, occ, occ_bwd = (d.cuda() if d is not None else None for d in (flow_fwd, occ, occ_bwd))
    mask = viewpoint_cam.motion_mask
    if dataset_type == "Hyper":
        mask = 1 - mask
    
    next_cam, prev_cam = viewpoint_cam.next_cam, viewpoint_cam.prev_cam

    if next_cam is not None:
        _, flow_bwd, _, _ =  next_cam.flow
        flow_bwd = flow_bwd.cuda() if flow_bwd is not None else None
        depth_next = next_cam.depth.unsqueeze(0).cuda()
        H, W = depth_next.shape[1:]
        uv_4, xyz_next = unproject(depth_next, next_cam.w2c.cuda(), next_cam.k.cuda()) #calculation for p4
        xyz_next = xyz_next.reshape(3, -1) #[3,H,W]->[3,HxW]
        uv_4 = uv_4.reshape(2, -1)         #[2,H,W]->[2,HxW]
        uv_2, _ = project(xyz_next, viewpoint_cam.w2c.cuda(), viewpoint_cam.k.cuda())  #calculation for p2
        
        # calculation of motion flow with motion mask
        camera_flow = (uv_2 - uv_4).reshape(2, H, W).permute(1, 2, 0).detach().cpu().numpy() # p2-p4
        flow_bwd = flow_bwd.detach().cpu().numpy() if flow_bwd is not None else None
        motion_flow_wo_mask = -(flow_bwd - camera_flow) #-[(p1 - p4) - (p2 - p4)] = p2 - p1 

        motion_flow_warp = warp_field(torch.from_numpy(motion_flow_wo_mask).to(flow_fwd), flow_fwd)
        motion_flow = motion_flow_warp.detach().cpu().numpy()
        motion_flow = motion_flow * np.transpose(mask, (1,2,0)) if mask is not None else motion_flow # [H,W,2](960, 536, 2)
        
        # calculation of gaussian flow
        time_next = next_cam.fid
        if isinstance(time_next, np.ndarray):
            time_next = torch.tensor(time_next, dtype=torch.float32)
        time_next = time_next.cuda()

        gs_per_pixel = render_pkg_re["gs_per_pixel"] #render_pkg["gs_per_pixel"]:[20,H,W] 
        gs_per_pixel = gs_per_pixel.permute(1, 2, 0).contiguous() #[H,W,20]
        weight_per_gs_pixel = render_pkg_re["weight_per_gs_pixel"]  # [20, H, W]
        proj_2D, conic_2D_inv, x_mu = render_pkg_re["proj_means_2D"], render_pkg_re["conic_2D_inv"], render_pkg_re["x_mu"]
        next_proj_2D, next_conic_2D = render_pkg_re_2["proj_means_2D"], render_pkg_re_2["conic_2D"]
        
        conic_2D_inv = conic_2D_inv.detach() # [K,3]
        gs_per_pixel = gs_per_pixel.long() # [K, H, W]
        
        cov2D_t_next_mtx = torch.zeros([next_conic_2D.shape[0], 2, 2]).cuda()
        cov2D_t_next_mtx[:, 0, 0] = next_conic_2D[:, 0]
        cov2D_t_next_mtx[:, 0, 1] = next_conic_2D[:, 1]
        cov2D_t_next_mtx[:, 1, 0] = next_conic_2D[:, 1]
        cov2D_t_next_mtx[:, 1, 1] = next_conic_2D[:, 2]
        
        cov2D_inv_t_present_mtx = torch.zeros([conic_2D_inv.shape[0], 2, 2]).cuda()
        cov2D_inv_t_present_mtx[:, 0, 0] = conic_2D_inv[:, 0]
        cov2D_inv_t_present_mtx[:, 0, 1] = conic_2D_inv[:, 1]
        cov2D_inv_t_present_mtx[:, 1, 0] = conic_2D_inv[:, 1]
        cov2D_inv_t_present_mtx[:, 1, 1] = conic_2D_inv[:, 2]
        
        U_t_2 = torch.svd(cov2D_t_next_mtx)[0]
        S_t_2 = torch.svd(cov2D_t_next_mtx)[1]
        V_t_2 = torch.svd(cov2D_t_next_mtx)[2]
        B_t_2 = torch.bmm(torch.bmm(U_t_2, torch.diag_embed(S_t_2)**(1/2)), V_t_2.transpose(1,2))
        
        U_inv_t_1 = torch.svd(cov2D_inv_t_present_mtx)[0]
        S_inv_t_1 = torch.svd(cov2D_inv_t_present_mtx)[1]
        V_inv_t_1 = torch.svd(cov2D_inv_t_present_mtx)[2]
        B_inv_t_1 = torch.bmm(torch.bmm(U_inv_t_1, torch.diag_embed(S_inv_t_1)**(1/2)), V_inv_t_1.transpose(1,2))
        
        B_t_2_B_inv_t_1 = torch.bmm(B_t_2, B_inv_t_1)

        conv_conv = torch.zeros([conic_2D_inv.shape[0], 2, 2], device=conic_2D_inv.device) # [K, 2, 2]
        conv_conv[:, 0, 0] = next_conic_2D[:, 0] * conic_2D_inv[:, 0] + next_conic_2D[:, 1] * conic_2D_inv[:, 1]
        conv_conv[:, 0, 1] = next_conic_2D[:, 0] * conic_2D_inv[:, 1] + next_conic_2D[:, 1] * conic_2D_inv[:, 2]
        conv_conv[:, 1, 0] = next_conic_2D[:, 1] * conic_2D_inv[:, 0] + next_conic_2D[:, 2] * conic_2D_inv[:, 1]
        conv_conv[:, 1, 1] = next_conic_2D[:, 1] * conic_2D_inv[:, 1] + next_conic_2D[:, 2] * conic_2D_inv[:, 2]
        conv_multi = (B_t_2_B_inv_t_1[gs_per_pixel].permute(2, 0, 1, 3, 4) @ x_mu.permute(0,2,3,1).unsqueeze(-1).detach()).squeeze() # K H W 2

        # Gaussian Flow
        motion_flow_per_pixel = (conv_multi + next_proj_2D[gs_per_pixel].permute(2,0,1,3) - proj_2D[gs_per_pixel].permute(2,0,1,3).detach() - x_mu.permute(0,2,3,1).detach()) # [K,H,W,2]
        # Velocity Flow
        next_proj_2D_velocity = render_pkg_re_velocity["proj_means_2D"]
        velocity_flow_per_pixel = (conv_multi + next_proj_2D_velocity[gs_per_pixel].permute(2,0,1,3) - proj_2D[gs_per_pixel].permute(2,0,1,3).detach() - x_mu.permute(0,2,3,1).detach()) # [K,H,W,2]

        weight_per_gs_pixel = weight_per_gs_pixel / (weight_per_gs_pixel.sum(dim=0, keepdim=True) + 1e-7) # [K,H,W]
        
        motion_flow_gs = torch.einsum("khw, khwa -> ahw", [weight_per_gs_pixel.detach(), motion_flow_per_pixel]) # [2,H,W]
        velocity_flow_gs = torch.einsum("khw, khwa -> ahw", [weight_per_gs_pixel.detach(), velocity_flow_per_pixel]) # [2,H,W]

        # Lagrangian Particle Flow Matching Loss
        motion_flow_t = torch.from_numpy(motion_flow).float().cuda()    # [H, W, 2]
        motion_gaussian_flow_t = motion_flow_gs.permute(1, 2, 0).float().cuda()
        velocity_gaussian_flow_t = velocity_flow_gs.permute(1, 2, 0).float().cuda()

        motion_flow_loss = calculate_flow_loss(motion_gaussian_flow_t, motion_flow_t.detach(), H, W)
        velocity_flow_loss = calculate_flow_loss(velocity_gaussian_flow_t, motion_flow_t.detach(), H, W)
        
        flow_loss = 0.5 * (motion_flow_loss + velocity_flow_loss)
        
        # saving gaussian flow, camera flow, motion flow
        flow_subdirs = [
            "gaussianflow",
            "cameraflow",
            "velocityflow",
            "motionflow_without_mask",
            "motionflow",
        ]
        
        required_dirs = [os.path.join(save_image_dir, "flow", sub) for sub in flow_subdirs]
        
        for d in required_dirs:
            os.makedirs(d, exist_ok=True)

        flow_paths = {
            name: os.path.join(save_image_dir, "flow", name, f"{iteration}.png")
            for name in flow_subdirs
        }

        gaussian_flow_path          = flow_paths["gaussianflow"]
        camera_flow_path            = flow_paths["cameraflow"]
        velocity_flow_path          = flow_paths["velocityflow"]
        motion_flow_without_mask_path = flow_paths["motionflow_without_mask"]
        motion_flow_path            = flow_paths["motionflow"]

        save_vis_flow_tofile(motion_gaussian_flow_t, gaussian_flow_path)
        save_vis_flow_tofile(camera_flow, camera_flow_path)
        save_vis_flow_tofile(velocity_gaussian_flow_t, velocity_flow_path)
        save_vis_flow_tofile(motion_flow_wo_mask, motion_flow_without_mask_path)
        save_vis_flow_tofile(motion_flow, motion_flow_path)
        
    return flow_loss

# almost same with motion_flow_velocity, used for render.py
def vis_flow(view, results, render_next, render_pkg_re_velocity):
    flow_fwd, _, occ, occ_bwd = view.flow
    flow_fwd, occ, occ_bwd = (d.cuda() if d is not None else None for d in (flow_fwd, occ, occ_bwd))
    mask = view.motion_mask
    next_cam = view.next_cam
    _, flow_bwd, _, _ =  next_cam.flow
    flow_bwd = flow_bwd.cuda() if flow_bwd is not None else None
    depth_next = next_cam.depth.unsqueeze(0).cuda()
    H, W = depth_next.shape[1:]
    uv_4, xyz_next = unproject(depth_next, next_cam.w2c.cuda(), next_cam.k.cuda())
    xyz_next = xyz_next.reshape(3, -1)
    uv_4 = uv_4.reshape(2, -1)
    uv_2, _ = project(xyz_next, view.w2c.cuda(), view.k.cuda())

    camera_flow = (uv_2 - uv_4).reshape(2, H, W).permute(1, 2, 0).detach().cpu().numpy()
    flow_bwd = flow_bwd.detach().cpu().numpy() if flow_bwd is not None else None
    motion_flow_wo_mask = -(flow_bwd - camera_flow)

    motion_flow_warp = warp_field(torch.from_numpy(motion_flow_wo_mask).to(flow_fwd), flow_fwd)
    motion_flow = motion_flow_warp.detach().cpu().numpy()
    motion_flow = motion_flow * np.transpose(mask, (1,2,0)) if mask is not None else motion_flow # [H,W,2](960, 536, 2)
    
    time_next = next_cam.fid
    if isinstance(time_next, np.ndarray):
        time_next = torch.tensor(time_next, dtype=torch.float32)
    time_next = time_next.cuda()

    gs_per_pixel = results["gs_per_pixel"]
    gs_per_pixel = gs_per_pixel.permute(1, 2, 0).contiguous()
    
    weight_per_gs_pixel = results["weight_per_gs_pixel"]
    proj_2D, conic_2D_inv, x_mu = results["proj_means_2D"], results["conic_2D_inv"], results["x_mu"]
    next_proj_2D, next_conic_2D = render_next["proj_means_2D"], render_next["conic_2D"]
    
    conic_2D_inv = conic_2D_inv.detach() # [K,3]
    gs_per_pixel = gs_per_pixel.long() # [K, H, W]
    
    cov2D_t_next_mtx = torch.zeros([next_conic_2D.shape[0], 2, 2]).cuda()
    cov2D_t_next_mtx[:, 0, 0] = next_conic_2D[:, 0]
    cov2D_t_next_mtx[:, 0, 1] = next_conic_2D[:, 1]
    cov2D_t_next_mtx[:, 1, 0] = next_conic_2D[:, 1]
    cov2D_t_next_mtx[:, 1, 1] = next_conic_2D[:, 2]
    
    cov2D_inv_t_present_mtx = torch.zeros([conic_2D_inv.shape[0], 2, 2]).cuda()
    cov2D_inv_t_present_mtx[:, 0, 0] = conic_2D_inv[:, 0]
    cov2D_inv_t_present_mtx[:, 0, 1] = conic_2D_inv[:, 1]
    cov2D_inv_t_present_mtx[:, 1, 0] = conic_2D_inv[:, 1]
    cov2D_inv_t_present_mtx[:, 1, 1] = conic_2D_inv[:, 2]
    
    U_t_2 = torch.svd(cov2D_t_next_mtx)[0]
    S_t_2 = torch.svd(cov2D_t_next_mtx)[1]
    V_t_2 = torch.svd(cov2D_t_next_mtx)[2]
    B_t_2 = torch.bmm(torch.bmm(U_t_2, torch.diag_embed(S_t_2)**(1/2)), V_t_2.transpose(1,2))
    
    U_inv_t_1 = torch.svd(cov2D_inv_t_present_mtx)[0]
    S_inv_t_1 = torch.svd(cov2D_inv_t_present_mtx)[1]
    V_inv_t_1 = torch.svd(cov2D_inv_t_present_mtx)[2]
    B_inv_t_1 = torch.bmm(torch.bmm(U_inv_t_1, torch.diag_embed(S_inv_t_1)**(1/2)), V_inv_t_1.transpose(1,2))
    
    B_t_2_B_inv_t_1 = torch.bmm(B_t_2, B_inv_t_1)

    conv_conv = torch.zeros([conic_2D_inv.shape[0], 2, 2], device=conic_2D_inv.device) # [K, 2, 2]
    conv_conv[:, 0, 0] = next_conic_2D[:, 0] * conic_2D_inv[:, 0] + next_conic_2D[:, 1] * conic_2D_inv[:, 1]
    conv_conv[:, 0, 1] = next_conic_2D[:, 0] * conic_2D_inv[:, 1] + next_conic_2D[:, 1] * conic_2D_inv[:, 2]
    conv_conv[:, 1, 0] = next_conic_2D[:, 1] * conic_2D_inv[:, 0] + next_conic_2D[:, 2] * conic_2D_inv[:, 1]
    conv_conv[:, 1, 1] = next_conic_2D[:, 1] * conic_2D_inv[:, 1] + next_conic_2D[:, 2] * conic_2D_inv[:, 2]
    conv_multi = (B_t_2_B_inv_t_1[gs_per_pixel].permute(2, 0, 1, 3, 4) @ x_mu.permute(0,2,3,1).unsqueeze(-1).detach()).squeeze()

    motion_flow_per_pixel = (conv_multi + next_proj_2D[gs_per_pixel].permute(2,0,1,3) - proj_2D[gs_per_pixel].permute(2,0,1,3).detach() - x_mu.permute(0,2,3,1).detach())
    next_proj_2D_velocity = render_pkg_re_velocity["proj_means_2D"]
    velocity_flow_per_pixel = (conv_multi + next_proj_2D_velocity[gs_per_pixel].permute(2,0,1,3) - proj_2D[gs_per_pixel].permute(2,0,1,3).detach() - x_mu.permute(0,2,3,1).detach())

    weight_per_gs_pixel = weight_per_gs_pixel / (weight_per_gs_pixel.sum(dim=0, keepdim=True) + 1e-7) # [K,H,W]
    
    motion_flow_gs = torch.einsum("khw, khwa -> ahw", [weight_per_gs_pixel.detach(), motion_flow_per_pixel]) # [2,H,W]
    motion_gaussian_flow_t = motion_flow_gs.permute(1, 2, 0).float().cuda()
    velocity_flow_gs = torch.einsum("khw, khwa -> ahw", [weight_per_gs_pixel.detach(), velocity_flow_per_pixel]) # [2,H,W]
    velocity_gaussian_flow_t = velocity_flow_gs.permute(1, 2, 0).float().cuda()
    
    return {"camera_flow": camera_flow,
            "motion_flow": motion_flow,
            "gaussian_flow": motion_gaussian_flow_t,
            "velocity_flow": velocity_gaussian_flow_t,
        }