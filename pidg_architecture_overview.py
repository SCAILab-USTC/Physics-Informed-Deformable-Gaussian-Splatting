# pidg_architecture_overview.py

"""
PIDG: Physics-Informed Deformable Gaussians - Architecture Overview

This file outlines the architecture of PIDG with emphasis on:
1. 4D Hash-based Deformable Field modeling
2. Static-Dynamic decoupled optimization
"""

import torch
from scene_PIDG import GaussianModel, DeformModel
from motion_utils.time_evolving_material_field import cauchy_momentum_constraint
from gaussian_renderer import render


# -------------------------------
# Core Model Initialization
# -------------------------------

def setup_pidg_models(dataset, opt):
    """
    Setup Gaussian model (particles) and Deformation model (4D hash MLP)
    """
    gaussians = GaussianModel(dataset.sh_degree)

    deform = DeformModel(
        grid_args=dataset.grid_args,                  # includes spatial/temporal hash encoder settings
        net_args=dataset.network_args,                # MLP architecture
        spatial_downsample_ratio=opt.spatial_downsample_ratio,
        spatial_perturb_range=opt.spatial_perturb_range,
        temporal_downsample_ratio=opt.temporal_downsample_ratio,
        temporal_perturb_range=opt.temporal_perturb_range,
        scale_xyz=dataset.scale_xyz,
        reg_spatial_able=opt.lambda_spatial_tv > 0.0,
        reg_temporal_able=opt.lambda_temporal_tv > 0.0,
    )

    return gaussians, deform


# -------------------------------
# Static-Dynamic Decoupling
# -------------------------------

def compute_dynamic_mask(scene, gaussians, deform):
    """
    Estimate dynamic points by projecting deformed Gaussians onto motion mask from multiple views.
    """
    dynamic_mask = init_dynamic_mask(scene, gaussians, deform)
    gaussians.dynamic_mask = dynamic_mask
    return dynamic_mask


# -------------------------------
# Deformation Application
# -------------------------------

def apply_deformation_for_iteration(gaussians, deform, fid, iteration, opt):
    """
    For each particle, apply 4D deformation based on dynamic mask and current frame time.
    """
    N = gaussians.get_xyz.shape[0]
    time_input = fid.unsqueeze(0).expand(N, -1)
    deform_pkgs = deform.step(gaussians.get_xyz, time_input)

    d_xyz = deform_pkgs['d_xyz']
    d_rotation = deform_pkgs['d_rotation']
    d_scaling = deform_pkgs['d_scaling']

    # Apply smoothing to static particles to maintain consistency
    if iteration > opt.densify_until_iter:
        mask = gaussians.dynamic_mask
        d_xyz[~mask] = 0.5 * d_xyz[~mask].detach() + 0.5 * d_xyz[~mask]
        d_rotation[~mask] = 0.5 * d_rotation[~mask].detach() + 0.5 * d_rotation[~mask]
        d_scaling[~mask] = 0.5 * d_scaling[~mask].detach() + 0.5 * d_scaling[~mask]

    return d_xyz, d_rotation, d_scaling


# -------------------------------
# Physics-Informed Loss Integration
# -------------------------------

def compute_physics_loss(gaussians, fid, total_frames, opt):
    """
    Compute Cauchy momentum loss from velocity and stress predicted by velocity_net
    """
    N = gaussians.get_xyz.shape[0]
    t_scalar = fid.float() / total_frames
    current_time = t_scalar.unsqueeze(0).expand(N, 1).requires_grad_(True)
    coords = torch.cat([gaussians.get_xyz, current_time], dim=1).requires_grad_(True)

    return cauchy_momentum_loss_sampled(
        gaussians, coords,
        dt=1 / total_frames,
        rho=1.0,
        sample_size=getattr(opt, 'physics_sample_size', 20000),
        chunk_size=getattr(opt, 'physics_chunk_size', 5000)
    )


# -------------------------------
# Rendering and Flow Matching
# -------------------------------

def render_with_deformation(viewpoint_cam, gaussians, deform, pipe, background, iteration, opt):
    """
    Core rendering function with deformation + optional motion flow supervision
    """
    d_xyz, d_rotation, d_scaling = apply_deformation_for_iteration(gaussians, deform, viewpoint_cam.fid, iteration, opt)
    render_pkg = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling)

    # Optional: if use_flow, compute flow-supervised rendering and loss
    if opt.use_flow and iteration > opt.densify_until_iter:
        # Compute deformation at next frame and do Lagrangian matching
        # Flow supervision logic here...
        pass

    return render_pkg


# -------------------------------
# Summary
# -------------------------------
"""
This architecture consists of:
- GaussianModel: holds 3D particles, opacity, SH features, and learned velocity/stress
- DeformModel: a 4D spatio-temporal hash-based deformation field predicting motion per-particle
- Dynamic mask: computed once and used to selectively apply deformation to moving Gaussians
- Training loop: alternates between rendering, loss computation (photometric + physics), and optimization

Optional: Optical flow loss and multi-frame velocity matching for dynamic consistency
"""

