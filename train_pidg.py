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

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from motion_utils.pidg import cauchy_momentum_constraint

def cauchy_momentum_loss_sampled(
    gaussians,
    coords,
    dt,
    rho,
    sample_size=20000,
    chunk_size=5000
):
    """
    1) Randomly sample 'sample_size' particles from N total particles;
    2) Split these sampled particles into chunks of size 'chunk_size' to compute the physics loss for each chunk;
    3) Accumulate the weighted loss proportionally and return the total.
    This avoids wasting compute power and prevents OOM issues.
    """
    N = coords.shape[0]
    # If N is smaller than the sample size, use all particles directly
    M = min(N, sample_size)
    # Randomly select M indices
    perm = torch.randperm(N, device=coords.device)[:M]
    coords_s = coords[perm]
    idx_s    = gaussians.particle_indices[perm]

    total_loss = 0.0
    # Compute in chunks
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
        # Weighted by (chunk_size / total_sample_size)
        total_loss += loss_chunk * ((end - start) / M)
        # Immediately clear PyTorch cache
        torch.cuda.empty_cache()

    # If sample_size < N, multiply total_loss by (N / M)
    # This scales it back to the full-particle average
    return total_loss * (M / N)

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

        # deformation and regularization
        reg = 0.0
        if iteration < opt.warm_up:
            d_rotation, d_scaling = 0.0, 0.0
            d_xyz = 0.0
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)
            xyz = gaussians.get_xyz.detach()
            deform_pkgs = deform.step(xyz, time_input)
            d_xyz, d_rotation, d_scaling = deform_pkgs['d_xyz'], deform_pkgs['d_rotation'], deform_pkgs['d_scaling']

            if opt.lambda_spatial_tv > 0.0 and (not opt.reg_after_densify or iteration > opt.densify_until_iter):
                reg += torch.mean(deform_pkgs['reg_spatial']) * opt.lambda_spatial_tv
            if opt.lambda_temporal_tv > 0.0 and (not opt.reg_after_densify or iteration > opt.densify_until_iter):
                reg += torch.mean(deform_pkgs['reg_temporal']) * opt.lambda_temporal_tv


        # Render
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + reg

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

            if iteration % 1000 == 0:
                # only for logging purpose
                # sample a fixed number of particles to log velocity and stress stats
                vs, ss = [], []
                perm2 = torch.randperm(N, device='cuda')[:1024]
                coords2 = coords[perm2]
                idx2    = gaussians.particle_indices[perm2]
                with torch.no_grad():
                    v2, s2 = gaussians.velocity_net(coords2, idx2)
                print(f"[Iter {iteration}] cauchy momentum loss = {cauchy_momentum_loss.item():.3e}")
                print(f"[Iter {iteration}] velocity mean = {v2.mean(dim=0).cpu().numpy()}")
                print(f"[Iter {iteration}] stress  mean = {s2.mean(dim=0).cpu().numpy()}")


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
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                                       testing_iterations, scene, render, (pipe, background), deform,
                                       dataset.load2gpu_on_the_fly)

            if iteration in testing_iterations:
                if cur_psnr.item() >= best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration
                    scene.save(iteration, True)
                    deform.save_weights(args.model_path, iteration, True)
                    print("Best: {} PSNR: {}".format(best_iteration, best_psnr))

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)

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
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    deform_pkgs = deform.step(xyz.detach(), time_input)
                    d_xyz, d_rotation, d_scaling = deform_pkgs['d_xyz'], deform_pkgs['d_rotation'], deform_pkgs['d_scaling']
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test.append(l1_loss(image, gt_image).mean().item())
                    psnr_test.append(psnr(image, gt_image).mean().item())

                l1_test = np.mean(l1_test)
                psnr_test = np.mean(psnr_test)
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


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
