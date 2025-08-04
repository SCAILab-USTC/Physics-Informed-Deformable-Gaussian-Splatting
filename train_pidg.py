import os
import torch
from random import randint, choice
from utils.loss_utils import l1_loss, ssim, kl_divergence
from gaussian_renderer import render, network_gui
import sys
from scene_PIDG import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, quaternion_to_matrix
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, merge_config
import numpy as np
from utils.flow_utils import motion_flow_velocity
import torchvision

from utils.loss_utils import ssim
from pytorch_msssim import ms_ssim
import contextlib
import lpips

with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
    lpips_fn = lpips.LPIPS(net='vgg').cuda()



try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from motion_utils.time_evolving_material_field import cauchy_momentum_constraint

def cauchy_momentum_loss_sampled(
    gaussians,
    coords,
    dt,
    rho,
    sample_size=20000,
    chunk_size=5000
):
    """
    1) Randomly draw sample_size points from N particles;
    2) Chunk these sampled points into pieces of size chunk_size and compute the physics loss per chunk;
    3) Accumulate them weighted by their proportion.
    This avoids wasted compute and OOM.
    """
    N = coords.shape[0]
    # If N is already smaller than sample_size, just use all of them
    M = min(N, sample_size)
    # Randomly sample M indices
    perm = torch.randperm(N, device=coords.device)[:M]
    coords_s = coords[perm]
    idx_s    = gaussians.particle_indices[perm]

    total_loss = 0.0
    # process in chunks
    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        c_chunk = coords_s[start:end]
        i_chunk = idx_s[start:end]
        loss_chunk = cauchy_momentum_constraint(
            c_chunk, 
            lambda x: gaussians.velocity_net(x, i_chunk),
            gaussians.force_sigma,
            dt, rho
        )
        # Weight by (chunk size / total sampled)
        total_loss += loss_chunk * ((end - start) / M)
        # Immediately free PyTorch cache
        torch.cuda.empty_cache()

    # If sample_size < N, scale total_loss back by M/N, equivalent to averaging over the full set
    return total_loss * (M / N)

def dilate_mask(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    assert kernel_size % 2 == 1
    padding = kernel_size // 2
    # Convert boolean mask to float (0 and 1)
    mask_float = mask.float().unsqueeze(0).unsqueeze(0)  # to [1, 1, H, W]
    # Create an all-ones dilation kernel
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32, device=mask.device)
    # Perform convolution (effectively checking if there's any True in the neighborhood)
    dilated = torch.nn.functional.conv2d(
        mask_float, 
        kernel, 
        padding=padding
    )
    dilated_mask = (dilated > 0).squeeze()
    
    return dilated_mask

def init_dynamic_mask(scene, gaussians, deform, batch_size=10001):
    device = "cuda"
    N = gaussians.get_xyz.shape[0]
    xyz_all = gaussians.get_xyz.detach()
    scores = torch.zeros(N, dtype=torch.float32, device=device)
    all_cameras = scene.getTrainCameras()
    camera_count = len(all_cameras)

    for cam in all_cameras:
        fid = cam.fid
        mask = cam.motion_mask.mean(0) > 0.
        mask = dilate_mask(torch.from_numpy(mask).cuda())
        H, W = mask.shape

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            xyz = xyz_all[start:end]
            time_input = fid.unsqueeze(0).expand(end - start, -1).to(device)
            deform_pkgs = deform.step(xyz, time_input)
            d_xyz = deform_pkgs['d_xyz'].detach()

            if torch.is_tensor(d_xyz):
                xyz_r = quaternion_to_matrix(d_xyz[..., :4])
                xyz_t = d_xyz[..., 4:, None]
                means3D = xyz_r @ xyz[..., None] + xyz_t
                means3D = means3D.squeeze(-1)
            else:
                means3D = xyz

            u, v = projectPoint(means3D, cam, H, W)
            valid_indices = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            in_mask = torch.full((end - start,), False, dtype=torch.bool, device=device)
            if valid_indices.any():
                in_mask[valid_indices] = mask[v[valid_indices], u[valid_indices]]
            scores[start:end][in_mask] += 1

    scores /= camera_count
    dynamic_mask = scores > 0.3
    print(f"Dynamic: {dynamic_mask.count_nonzero()} / {N}")
    return dynamic_mask

def projectPoint(xyz, cam, H, W):
    homogeneous_coords = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)
    clip_coords = torch.matmul(homogeneous_coords, cam.full_proj_transform)
    ndc_coords = clip_coords[:, :3] / clip_coords[:, 3:]
    u = (ndc_coords[:, 0] * 0.5 + 0.5) * W
    v = (ndc_coords[:, 1] * 0.5 + 0.5) * H
    u_int = torch.round(u).long()
    v_int = torch.round(v).long()
    return u_int, v_int


def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    tb_writer, args = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(
        grid_args=dataset.grid_args, 
        net_args=dataset.network_args,
        spatial_downsample_ratio=opt.spatial_downsample_ratio,
        spatial_perturb_range=opt.spatial_perturb_range, 
        temporal_downsample_ratio=opt.temporal_downsample_ratio,
        temporal_perturb_range=opt.temporal_perturb_range, 
        scale_xyz=dataset.scale_xyz,
        reg_spatial_able=opt.lambda_spatial_tv > 0.0,
        reg_temporal_able=opt.lambda_temporal_tv > 0.0,
    )
    deform.train_setting(opt)

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    for iteration in range(1, opt.iterations + 1):

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        if opt.data_sample == 'random':
            viewpoint_cam = choice(scene.getTrainCameras())
        elif opt.data_sample == 'order':
            viewpoint_cam = viewpoint_stack.pop(0)
        elif opt.data_sample == 'stack':
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid
        if iteration == opt.densify_until_iter:
            gaussians.dynamic_mask = init_dynamic_mask(scene, gaussians, deform)
        dynamic_mask = gaussians.dynamic_mask
        # deformation and regularization
        reg = 0.0
        N = gaussians.get_xyz.shape[0]
        d_xyz = torch.zeros((gaussians.get_xyz.shape[0], 7), device="cuda")
        d_rotation = torch.zeros((gaussians.get_xyz.shape[0], 4), device="cuda")
        d_scaling = torch.zeros_like(gaussians.get_scaling)
        if viewpoint_cam == None:
            print("mask Failed!")
        loss = 0.0
        if iteration < opt.warm_up:
            d_rotation, d_scaling = 0.0, 0.0
            d_xyz = 0.0
        else:
            N_dynamic = gaussians.dynamic_mask.count_nonzero()
            time_input = fid.unsqueeze(0).expand(N, -1)
            deform_pkgs = deform.step(gaussians.get_xyz, time_input)
        
            d_xyz = deform_pkgs['d_xyz']
            d_rotation = deform_pkgs['d_rotation']
            d_scaling = deform_pkgs['d_scaling']
            # Dynamic Particles
            if iteration > opt.densify_until_iter:
                d_xyz[~dynamic_mask] = 0.5 * deform_pkgs['d_xyz'][~dynamic_mask].detach() + 0.5 * deform_pkgs['d_xyz'][~dynamic_mask]
                d_rotation[~dynamic_mask] = 0.5 * deform_pkgs['d_rotation'][~dynamic_mask].detach() + 0.5 * deform_pkgs['d_rotation'][~dynamic_mask]
                d_scaling[~dynamic_mask] = 0.5 * deform_pkgs['d_scaling'][~dynamic_mask].detach() + 0.5 * deform_pkgs['d_scaling'][~dynamic_mask]
            if opt.lambda_spatial_tv > 0.0 and (not opt.reg_after_densify or iteration > opt.densify_until_iter):
                reg += torch.mean(deform_pkgs['reg_spatial']) * opt.lambda_spatial_tv
            if opt.lambda_temporal_tv > 0.0 and (not opt.reg_after_densify or iteration > opt.densify_until_iter):
                reg += torch.mean(deform_pkgs['reg_temporal']) * opt.lambda_temporal_tv
        # Render
        enable_flow_grad = True if opt.use_flow and (iteration % 100 == 0) and (iteration > opt.densify_until_iter) else False
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, enable_flow_grad)
        image, viewspace_point_tensor, visibility_filter, radii, depth = render_pkg_re["render"], render_pkg_re["viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"], render_pkg_re["depth"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        # mask = torch.from_numpy(viewpoint_cam.motion_mask).cuda()
        mask = 1.0 - torch.from_numpy(viewpoint_cam.motion_mask).cuda()
        
        gt_image_masked = gt_image * mask
        image_masked = image * mask
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + reg
        loss_static = (1.0 - opt.lambda_dssim) * l1_loss(image_masked, gt_image_masked) + opt.lambda_dssim * (1.0 - ssim(image_masked, gt_image_masked))
        if iteration > opt.densify_until_iter:
           loss += loss_static
           
        # Save depth and image   
        if  iteration % 50 == 0:
            depth = depth / (depth.max() + 1e-5)
            fname = f"iter_{iteration:06d}.png"
            depth_path = os.path.join(opt.save_image_dir, "depth", fname)
            image_path = os.path.join(opt.save_image_dir, "image", fname)
            os.makedirs(os.path.dirname(depth_path), exist_ok=True)
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            torchvision.utils.save_image(depth, depth_path)
            torchvision.utils.save_image(image, image_path)

        if iteration > opt.warm_up and (iteration % 100 == 0):
            total_frame = len(scene.getTrainCameras())
            t_scalar = fid.float() / total_frame
            N = gaussians.get_xyz.shape[0]
            current_time = t_scalar.unsqueeze(0).expand(N,1).requires_grad_(True)
            coords = torch.cat([gaussians.get_xyz, current_time], dim=1).requires_grad_(True)

            cauchy_momentum_loss = cauchy_momentum_loss_sampled(
                gaussians, coords,
                dt=1/total_frame,
                rho=1.0,
                sample_size=getattr(opt, 'physics_sample_size', 20000),
                chunk_size = getattr(opt, 'physics_chunk_size', 5000)
            )
            loss = loss + opt.physics_loss_weight * cauchy_momentum_loss
            
            # Calculation of Lagrangian Particle Flow Matching Loss
            if opt.use_flow and iteration > opt.densify_until_iter:
                if viewpoint_cam.next_cam is not None:
                    fid_next = viewpoint_cam.next_cam.fid
                    time_input_next = fid_next.unsqueeze(0).expand(N, -1)
                    deform_pkgs_next = deform.step(gaussians.get_xyz.detach(), time_input_next)
                    d_xyz_next, d_rotation_next, d_scaling_next = deform_pkgs_next['d_xyz'], deform_pkgs_next['d_rotation'], deform_pkgs_next['d_scaling']
                    enable_flow_grad = True
                    render_pkg_re_2 = render(viewpoint_cam, gaussians, pipe, background, d_xyz_next, d_rotation_next, d_scaling_next, enable_flow_grad)
                    velocity = gaussians.get_velocity_and_stress(viewpoint_cam.fid)[0]
                    xyz_t_velocity = d_xyz[..., 4:, None] + velocity.unsqueeze(2) * (viewpoint_cam.next_cam.fid - viewpoint_cam.fid) # [N, 3, 1]
                    d_xyz_velocity = torch.cat([d_xyz[..., :4], xyz_t_velocity.squeeze(-1)], dim=-1)
                    render_pkg_re_velocity = render(viewpoint_cam, gaussians, pipe, background, d_xyz_velocity, d_rotation, d_scaling, enable_flow_grad)
                    
                    flow_loss = motion_flow_velocity(iteration, gaussians, viewpoint_cam, render_pkg_re, render_pkg_re_2, render_pkg_re_velocity, opt.save_image_dir, opt.dataset_type)
                    
                    loss = loss + opt.flow_loss_weight * flow_loss
                    
                    print(iteration, "Optical flow loss = {:.3e}".format(flow_loss.item()))


            if iteration % 1000 == 0:
                perm2 = torch.randperm(N, device='cuda')[:1024]
                coords2 = coords[perm2]
                idx2    = gaussians.particle_indices[perm2]
                with torch.no_grad():
                    v2, s2 = gaussians.velocity_net(coords2, idx2)
                print(f"[Iter {iteration}] cauchy momentum loss = {cauchy_momentum_loss.item():.3e}")
                print(f"[Iter {iteration}] velocity mean = {v2.mean(dim=0).cpu().numpy()}")
                print(f"[Iter {iteration}] stress  mean = {s2.mean(dim=0).cpu().numpy()}")
                
        if iteration % 100 == 0 and iteration > opt.warm_up:
            true_count = dynamic_mask.sum().item()

            total_count = dynamic_mask.numel()


            # Ratio of Dynamic
            ratio = true_count / total_count
            print(f"Total Gaussians: {total_count} \nDynamic: {true_count}")
            print(f"Ratio: {ratio}")

        loss.backward()
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "pts": len(gaussians.get_xyz),
                    "reg": f"{reg:.{5}f}",
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # Log and save
            cur_psnr, cur_ssim, cur_ms_ssim, cur_lpips = training_report(
    tb_writer, iteration, Ll1, loss, l1_loss,
    testing_iterations, scene, render, (pipe, background), deform,
    dataset.load2gpu_on_the_fly
)


            if iteration in testing_iterations:
                if cur_psnr >= best_psnr:
                    best_psnr = cur_psnr
                    best_iteration = iteration
                    scene.save(iteration, True)
                    deform.save_weights(args.model_path, iteration, True)
                    gaussians.velocity_net.save_weights(args.model_path, iteration, True)
                    print("Best [{}]: PSNR: {:.2f}, SSIM: {:.4f}, MS-SSIM: {:.4f}, LPIPS: {:.4f}".format(
                        best_iteration, best_psnr, cur_ssim, cur_ms_ssim, cur_lpips
                    ))


            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)
                gaussians.velocity_net.save_weights(args.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, opt.disable_ws_prune)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer, args


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)

    test_psnr = 0.0
    test_ssim = 0.0
    test_ms_ssim = 0.0
    test_lpips = 0.0 

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = []
                psnr_test = []
                ssim_test_list = []
                ms_ssim_test_list = []
                lpips_test_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    deform_pkgs = deform.step(xyz.detach(), time_input)
                    d_xyz, d_rotation, d_scaling = deform_pkgs['d_xyz'], deform_pkgs['d_rotation'], deform_pkgs['d_scaling']
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, False)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    image_batched = image.unsqueeze(0)
                    gt_image_batched = gt_image.unsqueeze(0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                    # Metrics
                    l1_test.append(l1_loss(image, gt_image).mean().item())
                    psnr_test.append(psnr(image, gt_image).mean().item())
                    ssim_test_list.append(ssim(image, gt_image).mean().item())
                    ms_ssim_test_list.append(ms_ssim(image_batched, gt_image_batched, data_range=1.0).item())
                    lpips_val = lpips_fn(image_batched, gt_image_batched).mean().item()
                    lpips_test_list.append(lpips_val)

                l1_avg = np.mean(l1_test)
                psnr_avg = np.mean(psnr_test)
                ssim_avg = np.mean(ssim_test_list)
                ms_ssim_avg = np.mean(ms_ssim_test_list)
                lpips_avg = np.mean(lpips_test_list)

                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_avg
                    test_ssim = ssim_avg
                    test_ms_ssim = ms_ssim_avg
                    test_lpips = lpips_avg

                print("\n[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.2f} SSIM {:.4f} MS-SSIM {:.4f} LPIPS {:.4f}".format(
                    iteration, config['name'], l1_avg, psnr_avg, ssim_avg, ms_ssim_avg, lpips_avg))

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_avg, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_avg, iteration)
                    tb_writer.add_scalar(config['name'] + '/ssim', ssim_avg, iteration)
                    tb_writer.add_scalar(config['name'] + '/ms_ssim', ms_ssim_avg, iteration)
                    tb_writer.add_scalar(config['name'] + '/lpips', lpips_avg, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr, test_ssim, test_ms_ssim, test_lpips


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--conf', type=str, default=None)
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,# default=[])
                       default=[5000, 6000, 7_000] + list(range(10000, 50001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5000, 10000, 20000, 30000, 40000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    if args.conf is not None and os.path.exists(args.conf):
        print("Find Config:", args.conf)
        args = merge_config(args, args.conf)
    else:
        print("[WARNING] Using default config.")

    # Initialize system state (RNG)
    safe_state(args.quiet)
    args.data_device = "cuda:0" if args.data_device == 'cuda' else args.data_device
    torch.cuda.set_device(args.data_device)

    if not args.quiet:
        print(vars(args))

    # Configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)

    # All done
    print("\nTraining complete.")
