import torch
from scene_PIDG import Scene, DeformModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state, quaternion_to_matrix
from utils.pose_utils import pose_spherical, render_wander_path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, merge_config
from gaussian_renderer import GaussianModel
import imageio
import numpy as np
import time
from PIL import Image
from utils.flow_vis_utils import save_vis_flow_tofile
from utils.flow_utils import vis_flow

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

def init_dynamic_mask(all_cameras, gaussians, deform, batch_size=10001):
    device = "cuda"
    N = gaussians.get_xyz.shape[0]
    xyz_all = gaussians.get_xyz.detach()
    scores = torch.zeros(N, dtype=torch.float32, device=device)
    camera_count = len(all_cameras)

    for cam in all_cameras:
        fid = cam.fid
        mask = cam.motion_mask.mean(0) > 0.
        mask = dilate_mask(mask)
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
    dynamic_mask = scores > 0.4
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


def render_set(model_path, load2gpu_on_the_fly, name, iteration, views, gaussians, pipeline, background, deform):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    camera_flow_path = os.path.join(model_path, name, "ours_{}".format(iteration), "camera_flow")
    motion_flow_path = os.path.join(model_path, name, "ours_{}".format(iteration), "motion_flow")
    gaussian_flow_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gaussian_flow")
    velocity_flow_path = os.path.join(model_path, name, "ours_{}".format(iteration), "velocity_flow")
    speed_path = os.path.join(model_path, name, "ours_{}".format(iteration), "speed.txt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(camera_flow_path, exist_ok=True)
    makedirs(motion_flow_path, exist_ok=True)
    makedirs(gaussian_flow_path, exist_ok=True)
    makedirs(velocity_flow_path, exist_ok=True)

    total_time = 0.0
    mask = gaussians.dynamic_mask
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpu_on_the_fly:
            view.load2device()
        fid = view.fid
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        t = time.time()
        d_xyz = torch.zeros((gaussians.get_xyz.shape[0], 7), device="cuda")
        d_xyz[:, 0] = 1.0
        d_rotation = torch.zeros((gaussians.get_xyz.shape[0], 4), device="cuda")
        d_scaling = torch.zeros_like(gaussians.get_scaling)
        if True:
            deform_pkgs = deform.step(xyz.detach(), time_input, fixed_attention=True)
            d_xyz = deform_pkgs['d_xyz']
            d_rotation = deform_pkgs['d_rotation']
            d_scaling = deform_pkgs['d_scaling']

            '''
            d_xyz[mask] = deform_pkgs['d_xyz']
            d_rotation[mask] = deform_pkgs['d_rotation']
            d_scaling[mask] = deform_pkgs['d_scaling']
            '''
        '''
        d_xyz = d_xyz * dynamic_mask.float().unsqueeze(-1)
        d_rotation = d_rotation * dynamic_mask.float().unsqueeze(-1)
        d_scaling = d_scaling * dynamic_mask.float().unsqueeze(-1)  
        '''
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, False)
        total_time += time.time() - t
        rendering = results["render"]
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)
    
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        
        if name == "train" and view.next_cam is not None:
            fid_next = view.next_cam.fid
            time_input_next = fid_next.unsqueeze(0).expand(gaussians.get_xyz.shape[0], -1)
            deform_pkgs_next = deform.step(gaussians.get_xyz.detach(), time_input_next)
            d_xyz_next, d_rotation_next, d_scaling_next = deform_pkgs_next['d_xyz'], deform_pkgs_next['d_rotation'], deform_pkgs_next['d_scaling']
            
            render_next = render(view, gaussians, pipeline, background, d_xyz_next, d_rotation_next, d_scaling_next, False)
            velocity, _ = gaussians.get_velocity_and_stress(view.fid)
            xyz_t_velocity = d_xyz[..., 4:, None] + velocity.unsqueeze(2) * (view.next_cam.fid - view.fid) # [N, 3, 1]
            d_xyz_velocity = torch.cat([d_xyz[..., :4], xyz_t_velocity.squeeze(-1)], dim=-1)
            render_pkg_re_velocity = render(view, gaussians, pipeline, background, d_xyz_velocity, d_rotation, d_scaling, False)
            flow_all = vis_flow(view, results, render_next, render_pkg_re_velocity)
            save_vis_flow_tofile(flow_all["camera_flow"], os.path.join(camera_flow_path, '{0:05d}'.format(idx) + ".png"))
            save_vis_flow_tofile(flow_all["motion_flow"], os.path.join(motion_flow_path, '{0:05d}'.format(idx) + ".png"))
            save_vis_flow_tofile(flow_all["gaussian_flow"], os.path.join(gaussian_flow_path, '{0:05d}'.format(idx) + ".png"))
            save_vis_flow_tofile(flow_all["velocity_flow"], os.path.join(velocity_flow_path, '{0:05d}'.format(idx) + ".png"))

    fps = len(views) / total_time
    print("FPS:", fps)
    with open(speed_path, "w") as f:
        f.write("FPS: " + str(fps))

def interpolate_time(model_path, load2gpt_on_the_fly, name, iteration, views, gaussians, pipeline, background, deform):
    render_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    frame = 150
    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]
    renderings = []
    for t in tqdm(range(0, frame, 1), desc="Rendering progress"):
        fid = torch.Tensor([t / (frame - 1)]).cuda()
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        deform_pkgs = deform.step(xyz.detach(), time_input)
        d_xyz, d_rotation, d_scaling = deform_pkgs['d_xyz'], deform_pkgs['d_rotation'], deform_pkgs['d_scaling']
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(t) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(t) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_view(model_path, load2gpt_on_the_fly, name, iteration, views, gaussians, pipeline, background, timer):
    render_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "depth")
    # acc_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "acc")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    # makedirs(acc_path, exist_ok=True)

    frame = 150
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]  # Choose a specific time for rendering

    render_poses = torch.stack(render_wander_path(view), 0)
    # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]],
    #                            0)

    renderings = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = view.fid

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        deform_pkgs = timer.step(xyz.detach(), time_input)
        d_xyz, d_rotation, d_scaling = deform_pkgs['d_xyz'], deform_pkgs['d_rotation'], deform_pkgs['d_scaling']
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_all(model_path, load2gpt_on_the_fly, name, iteration, views, gaussians, pipeline, background, deform):
    render_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = 150
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]],
                               0)
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]  # Choose a specific time for rendering

    renderings = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = torch.Tensor([i / (frame - 1)]).cuda()

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        deform_pkgs = deform.step(xyz.detach(), time_input)
        d_xyz, d_rotation, d_scaling = deform_pkgs['d_xyz'], deform_pkgs['d_rotation'], deform_pkgs['d_scaling']
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_poses(model_path, load2gpt_on_the_fly, name, iteration, views, gaussians, pipeline, background, timer):
    render_path = os.path.join(model_path, name, "interpolate_pose_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_pose_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    # makedirs(acc_path, exist_ok=True)
    frame = 520
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view_begin = views[0]  # Choose a specific time for rendering
    view_end = views[-1]
    view = views[idx]

    R_begin = view_begin.R
    R_end = view_end.R
    t_begin = view_begin.T
    t_end = view_end.T

    renderings = []
    for i in tqdm(range(frame), desc="Rendering progress"):
        fid = view.fid

        ratio = i / (frame - 1)

        R_cur = (1 - ratio) * R_begin + ratio * R_end
        T_cur = (1 - ratio) * t_begin + ratio * t_end

        view.reset_extrinsic(R_cur, T_cur)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        deform_pkgs = timer.step(xyz.detach(), time_input)
        d_xyz, d_rotation, d_scaling = deform_pkgs['d_xyz'], deform_pkgs['d_rotation'], deform_pkgs['d_scaling']

        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=60, quality=8)


def interpolate_view_original(model_path, load2gpt_on_the_fly, name, iteration, views, gaussians, pipeline, background,
                              timer):
    render_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "depth")
    # acc_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "acc")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = 1000
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    R = []
    T = []
    for view in views:
        R.append(view.R)
        T.append(view.T)

    view = views[0]
    renderings = []
    for i in tqdm(range(frame), desc="Rendering progress"):
        fid = torch.Tensor([i / (frame - 1)]).cuda()

        query_idx = i / frame * len(views)
        begin_idx = int(np.floor(query_idx))
        end_idx = int(np.ceil(query_idx))
        if end_idx == len(views):
            break
        view_begin = views[begin_idx]
        view_end = views[end_idx]
        R_begin = view_begin.R
        R_end = view_end.R
        t_begin = view_begin.T
        t_end = view_end.T

        ratio = query_idx - begin_idx

        R_cur = (1 - ratio) * R_begin + ratio * R_end
        T_cur = (1 - ratio) * t_begin + ratio * t_end

        view.reset_extrinsic(R_cur, T_cur)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        deform_pkgs = timer.step(xyz.detach(), time_input)
        d_xyz, d_rotation, d_scaling = deform_pkgs['d_xyz'], deform_pkgs['d_rotation'], deform_pkgs['d_scaling']

        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                mode: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        deform = DeformModel(dataset.grid_args, dataset.network_args, scale_xyz=dataset.scale_xyz, reg_spatial_able=False, reg_temporal_able=False)
        deform.load_weights(dataset.model_path, iteration=iteration)
        gaussians.velocity_net.load_weights(dataset.model_path, iteration=iteration)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if mode == "render":
            render_func = render_set
        elif mode == "time":
            render_func = interpolate_time
        elif mode == "view":
            render_func = interpolate_view
        elif mode == "pose":
            render_func = interpolate_poses
        elif mode == "original":
            render_func = interpolate_view_original
        else:
            render_func = interpolate_all

        if not skip_train:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, "train", scene.loaded_iter,
                        scene.getTrainCameras(), gaussians, pipeline,
                        background, deform)

        if not skip_test:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, "test", scene.loaded_iter,
                        scene.getTestCameras(), gaussians, pipeline,
                        background, deform)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument('--conf', type=str, default=None)
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original'])
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    if args.conf is not None and os.path.exists(args.conf):
        print("Find Config:", args.conf)
        args = merge_config(args, args.conf)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    args.data_device = 'cuda:0' if args.data_device == 'cuda' else args.data_device
    torch.cuda.set_device(args.data_device)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.mode)
