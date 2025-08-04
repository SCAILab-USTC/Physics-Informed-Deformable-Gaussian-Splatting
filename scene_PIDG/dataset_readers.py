import torch
import os
import sys
from PIL import Image
from typing import NamedTuple, Optional
from scene_PIDG.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
import imageio
from glob import glob
import cv2 as cv
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene_PIDG.gaussian_model import BasicPointCloud
from utils.camera_utils import camera_nerfies_from_JSON
from tqdm import tqdm


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    depth: Optional[np.array] = None
    mask_image: Optional[np.array] = None
    flow: Optional[list] = None  # [flow_fwd, flow_bwd, occ_fwd, occ_bwd]


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]]
                 for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return K, pose


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    num_frames = len(cam_extrinsics)
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write(
            "Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        fid = int(image_name) / (num_frames - 1)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, fid=fid)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'],
                       vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile_train, transformsfile_test, white_background, extension=".png"):
    train_json = json.load(open(os.path.join(path, transformsfile_train)))
    test_json  = json.load(open(os.path.join(path, transformsfile_test)))
    fovx_train   = train_json["camera_angle_x"]
    frames_train = train_json["frames"]
    fovx_test    = test_json.get("camera_angle_x", fovx_train)
    frames_test  = test_json["frames"]
    
    train_ids = [f["file_path"] for f in frames_train]
    test_ids  = [f["file_path"] for f in frames_test]

    cam_info_dict = {}
    def build_cam_info(frame, fovx, is_train):
        file_id   = frame["file_path"]                # e.g. "train/0000"
        rel_path  = file_id + extension               # e.g. "train/0000.png"
        image_path = os.path.join(path, rel_path)
        image_name = Path(rel_path).stem              # e.g. "0000"

        # Read Oringinal Image and choose white/black background
        image_rgba = Image.open(image_path).convert("RGBA")
        im_data = np.array(image_rgba) / 255.0
        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
        alpha = im_data[..., 3:4]
        rgb   = im_data[..., :3] * alpha + bg * (1 - alpha)
        image = Image.fromarray((rgb * 255.0).astype(np.uint8), "RGB")

        # Camera Extrinsics
        matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
        R = -matrix[:3, :3].T
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        # FoV
        fovy = focal2fov(fov2focal(fovx, image.width), image.height)
        FovY, FovX = fovx, fovy

        #If train，read flow / occ / mask / depth
        flow_fwd = flow_bwd = occ_fwd = occ_bwd = None
        mask_np  = None
        depth_tensor = None

        if is_train:
            # forward/backward flow
            from utils.flow_utils import readFlow
            flow_dir = os.path.join(path, "flows_flo")
            fwd_flow_file = os.path.join(flow_dir, f"flow_fwd_{image_name}.flo")
            if os.path.isfile(fwd_flow_file):
                flow_fwd = torch.from_numpy(readFlow(fwd_flow_file)).float().contiguous()
            bwd_flow_file = os.path.join(flow_dir, f"flow_bwd_{image_name}.flo")
            if os.path.isfile(bwd_flow_file):
                flow_bwd = torch.from_numpy(readFlow(bwd_flow_file)).float().contiguous()

            # occulusion masks
            occ_fwd_path = os.path.join(flow_dir, f"occ_fwd_{image_name}.png")
            if os.path.isfile(occ_fwd_path):
                occ_fwd_np = (np.array(Image.open(occ_fwd_path)) != 0).astype(np.uint8)
                occ_fwd = torch.from_numpy(occ_fwd_np).contiguous()

            occ_bwd_path = os.path.join(flow_dir, f"occ_bwd_{image_name}.png")
            if os.path.isfile(occ_bwd_path):
                occ_bwd_np = (np.array(Image.open(occ_bwd_path)) != 0).astype(np.uint8)
                occ_bwd = torch.from_numpy(occ_bwd_np).contiguous()

            # sam mask
            mask_path  = os.path.join(path, "resized_mask", image_name + extension)
            if os.path.isfile(mask_path):
                mask_gray = np.array(Image.open(mask_path).convert("L"))
                mask_bin  = (mask_gray > 128).astype(np.uint8)[..., None]  # [H,W,1]
                mask_np   = mask_bin # [1,H,W]

            # depth
            depth_path = os.path.join(path, "depth_distill", image_name + ".npy")
            if os.path.isfile(depth_path):
                depth_tensor = torch.from_numpy(np.load(depth_path)).float()

        cam_info = CameraInfo(
            uid        = file_id,
            R          = R,
            T          = T,
            FovY       = FovY,
            FovX       = FovX,
            image      = image,
            image_path = image_path,
            image_name = image_name,
            width      = image.width,
            height     = image.height,
            fid        = frame['time'],
            flow       = [flow_fwd, flow_bwd, occ_fwd, occ_bwd] if is_train else None,
            mask_image = mask_np if is_train else None,
            depth      = depth_tensor if is_train else None
        )
        return cam_info

    #Read train first, then test
    for f in frames_train:
        cam_info_dict[f["file_path"]] = build_cam_info(f, fovx_train, is_train=True)
    for f in frames_test:
        cam_info_dict[f["file_path"]] = build_cam_info(f, fovx_test,  is_train=False)

    #Build train_cam_infos_all (next/previous camera built on train ids, avoiding leaking)
    train_cam_infos = []
    for i, fid in enumerate(train_ids):
        prev_id = train_ids[i-1] if i > 0 else None
        next_id = train_ids[i+1] if i < len(train_ids) - 1 else None

        train_cam_infos.append({
            "cam":      cam_info_dict[fid],
            "cam_prev": cam_info_dict[prev_id] if prev_id else None,
            "cam_next": cam_info_dict[next_id] if next_id else None,
        })

    #Build test_cam_infos
    test_cam_infos = [cam_info_dict[fid] for fid in test_ids]


    return train_cam_infos, test_cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms and Test Transforms")
    train_cam_infos, test_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    train_cams = [entry["cam"] for entry in train_cam_infos]
    nerf_normalization = getNerfppNorm(train_cams)
    
    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readDTUCameras(path, render_camera, object_camera):
    camera_dict = np.load(os.path.join(path, render_camera))
    images_lis = sorted(glob(os.path.join(path, 'image/*.png')))
    masks_lis = sorted(glob(os.path.join(path, 'mask/*.png')))
    n_images = len(images_lis)
    cam_infos = []
    cam_idx = 0
    for idx in range(0, n_images):
        image_path = images_lis[idx]
        image = np.array(Image.open(image_path))
        mask = np.array(imageio.imread(masks_lis[idx])) / 255.0
        image = Image.fromarray((image * mask).astype(np.uint8))
        world_mat = camera_dict['world_mat_%d' % idx].astype(np.float32)
        fid = camera_dict['fid_%d' % idx] / (n_images / 12 - 1)
        image_name = Path(image_path).stem
        scale_mat = camera_dict['scale_mat_%d' % idx].astype(np.float32)
        P = world_mat @ scale_mat
        P = P[:3, :4]

        K, pose = load_K_Rt_from_P(None, P)
        a = pose[0:1, :]
        b = pose[1:2, :]
        c = pose[2:3, :]

        pose = np.concatenate([a, -c, -b, pose[3:, :]], 0)

        S = np.eye(3)
        S[1, 1] = -1
        S[2, 2] = -1
        pose[1, 3] = -pose[1, 3]
        pose[2, 3] = -pose[2, 3]
        pose[:3, :3] = S @ pose[:3, :3] @ S

        a = pose[0:1, :]
        b = pose[1:2, :]
        c = pose[2:3, :]

        pose = np.concatenate([a, c, b, pose[3:, :]], 0)

        pose[:, 3] *= 0.5

        matrix = np.linalg.inv(pose)
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        FovY = focal2fov(K[0, 0], image.size[1])
        FovX = focal2fov(K[0, 0], image.size[0])
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=image.size[
                                  0], height=image.size[1],
                              fid=fid)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readNeuSDTUInfo(path, render_camera, object_camera):
    print("Reading DTU Info")
    train_cam_infos = readDTUCameras(path, render_camera, object_camera)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=[],
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readNerfiesCameras(path):
    with open(f'{path}/scene.json', 'r') as f:
        scene_json = json.load(f)
    with open(f'{path}/metadata.json', 'r') as f:
        meta_json = json.load(f)
    with open(f'{path}/dataset.json', 'r') as f:
        dataset_json = json.load(f)

    coord_scale  = scene_json['scale']
    scene_center = scene_json['center']

    if 'vrig' in path:
        train_img = dataset_json['train_ids']
        val_img   = dataset_json['val_ids']
        all_img   = train_img + val_img
        ratio     = 0.5
    elif 'interp' in path:
        all_id    = dataset_json['ids']
        train_img = all_id[::4]
        val_img   = all_id[2::4]
        all_img   = train_img + val_img
        ratio     = 0.5
    elif 'nerf' in path:
        train_img = dataset_json['train_ids']
        val_img   = dataset_json['val_ids']
        all_img   = train_img + val_img
        ratio     = 1.0
        print("Assuming NeRF-DS dataset!")
    else:  # hypernerf default
        train_img = dataset_json['ids'][::4]
        val_img   = []
        all_img   = train_img
        ratio     = 0.5

    all_cam_params = []
    for im in all_img:
        cam_json = f'{path}/camera/{im}.json'
        camera = camera_nerfies_from_JSON(cam_json, ratio)
        camera['position'] = (camera['position'] - scene_center) * coord_scale
        all_cam_params.append(camera)

    scale_folder = f'{int(1/ratio)}x'
    all_img_paths = [f'{path}/rgb/{scale_folder}/{i}.png' for i in all_img]

    all_time_raw = [meta_json[i]['time_id'] for i in all_img]
    max_time     = max(all_time_raw) if len(all_time_raw) > 0 else 1
    all_time     = [t / max_time for t in all_time_raw]

    cam_infos = []
    # id->index
    id_to_index = {}

    for idx, (img_id, img_path) in enumerate(zip(all_img, all_img_paths)):
        image_np  = np.array(Image.open(img_path))
        image     = Image.fromarray(image_np.astype(np.uint8))
        image_name = Path(img_path).stem
        fid       = all_time[idx]

        orientation = all_cam_params[idx]['orientation'].T
        position    = -all_cam_params[idx]['position'] @ orientation
        T = position
        R = orientation
        focal = all_cam_params[idx]['focal_length']
        FovY  = focal2fov(focal, image.size[1])
        FovX  = focal2fov(focal, image.size[0])

        if img_id in val_img:
            cam_info = CameraInfo(
                uid=idx, R=R, T=T,
                FovY=FovY, FovX=FovX,
                image=image, image_path=img_path,
                image_name=image_name,
                width=image.size[0], height=image.size[1],
                fid=fid,
            )
            cam_infos.append(cam_info)
            id_to_index[img_id] = idx
            continue

        #If train，read flow / occ / mask / depth
        flow_fwd = flow_bwd = occ_fwd = occ_bwd = None
        mask_np = None
        depth_tensor = None
        if img_id in train_img:
            flow_fwd = flow_bwd = occ_fwd = occ_bwd = None
            mask_np = None
            depth_tensor = None

            flow_dir = os.path.join(path, "flow", "2x")
            flow_ext = "flo"

            flow_fwd_path = os.path.join(flow_dir, f"{image_name}_flow_fwd.{flow_ext}")
            from utils.flow_utils import readFlow
            if os.path.isfile(flow_fwd_path):
                flow_fwd_np = readFlow(flow_fwd_path)  # [H,W,2]
                flow_fwd = torch.from_numpy(flow_fwd_np).float().contiguous()
                occ_fwd_path = os.path.join(flow_dir, f"{image_name}_occ_fwd.png")
                if os.path.isfile(occ_fwd_path):
                    occ_fwd_np = (np.array(Image.open(occ_fwd_path)) != 0).astype(np.uint8)
                    occ_fwd = torch.from_numpy(occ_fwd_np).contiguous()

            flow_bwd_path = os.path.join(flow_dir, f"{image_name}_flow_bwd.{flow_ext}")
            if os.path.isfile(flow_bwd_path):
                flow_bwd_np = readFlow(flow_bwd_path)
                flow_bwd = torch.from_numpy(flow_bwd_np).float().contiguous()
                occ_bwd_path = os.path.join(flow_dir, f"{image_name}_occ_bwd.png")
                if os.path.isfile(occ_bwd_path):
                    occ_bwd_np = (np.array(Image.open(occ_bwd_path)) != 0).astype(np.uint8)
                    occ_bwd = torch.from_numpy(occ_bwd_np).contiguous()

            #SAM mask
            sam_mask_dir = os.path.join(path, "resized_mask", "2x")
            sam_mask_path = os.path.join(sam_mask_dir, image_name + ".png")
            if os.path.isfile(sam_mask_path):
                mask_gray = np.array(Image.open(sam_mask_path).convert("L"))
                mask_bin  = (mask_gray > 128).astype(np.uint8)[..., None]  # [H,W,1]
                mask_np   = np.transpose(mask_bin, (2,0,1))  # 保持 numpy

            #Depth
            depth_dir  = os.path.join(path, "depth_distill", "2x")
            depth_path = os.path.join(depth_dir, image_name + ".npy")
            if os.path.isfile(depth_path):
                depth_arr = np.load(depth_path)  # [H,W] or [H,W,1]
                depth_tensor = torch.from_numpy(depth_arr).float()
                
            cam_info = CameraInfo(
                uid=idx, R=R, T=T,
                FovY=FovY, FovX=FovX,
                image=image, image_path=img_path,
                image_name=image_name,
                width=image.size[0], height=image.size[1],
                fid=fid,flow=[flow_fwd,flow_bwd,occ_fwd,occ_bwd],mask_image=mask_np,depth=depth_tensor
            )
            cam_infos.append(cam_info)
            id_to_index[img_id] = idx

    timeline = list(all_img)

    #Build train_cam_infos_all (next/previous camera built on train ids, avoiding leaking)
    train_cam_infos_all = []
    for img_id in train_img:
        cur_idx = id_to_index[img_id]
        pos     = timeline.index(img_id)
        prev_id = timeline[pos - 1] if pos > 0 else None
        next_id = timeline[pos + 1] if pos < len(timeline) - 1 else None
        if prev_id not in train_img:
            prev_id = None
        if next_id not in train_img:
            next_id = None
        prev_cam = cam_infos[id_to_index[prev_id]] if prev_id else None
        next_cam = cam_infos[id_to_index[next_id]] if next_id else None

        train_cam_infos_all.append({
            "cam":      cam_infos[cur_idx],
            "cam_prev": prev_cam,
            "cam_next": next_cam,
        })

    #Build test_cam_infos_all
    test_cam_infos_all = [cam_infos[id_to_index[i]] for i in val_img]

    return cam_infos, train_cam_infos_all, test_cam_infos_all, scene_center, coord_scale

def readNerfiesInfo(path, eval):
    print("Reading Nerfies Info")
    cam_infos, train_cam_infos_all, test_cam_infos_all, scene_center, scene_scale = readNerfiesCameras(path)

    if eval:
        train_cam_infos = train_cam_infos_all
        test_cam_infos = test_cam_infos_all
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    if isinstance(train_cam_infos[0], dict):
        # train_cam_infos 是 list of dict
        temp_cam_list = [ item["cam"] for item in train_cam_infos ]
    else:
        temp_cam_list = train_cam_infos
        
    nerf_normalization = getNerfppNorm(temp_cam_list)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        print(f"Generating point cloud from nerfies...")

        xyz = np.load(os.path.join(path, "points.npy"))
        xyz = (xyz - scene_center) * scene_scale
        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        print("Find sfm point cloud:", ply_path)

    try:
        pcd = fetchPly(ply_path)
        print("Load sfm point cloud from:", ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromNpy(path, npy_file, split, hold_id, num_images):
    cam_infos = []
    video_paths = sorted([a for a in glob(os.path.join(path, '*')) if os.path.isdir(a)])
    poses_bounds = np.load(os.path.join(path, npy_file))

    poses = poses_bounds[:, :15].reshape(-1, 3, 5)
    H, W, focal = poses[0, :, -1]

    n_cameras = poses.shape[0]
    poses = np.concatenate(
        [poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    
    bottoms = np.array([0, 0, 0, 1]).reshape(
        1, -1, 4).repeat(poses.shape[0], axis=0)
    poses = np.concatenate([poses, bottoms], axis=1)
    poses = poses @ np.diag([1, -1, -1, 1])

    i_test = np.array(hold_id)
    video_list = i_test if split != 'train' else list(
        set(np.arange(n_cameras)) - set(i_test))

    for i in video_list:
        video_path = os.path.join(video_paths[i], "images")
        c2w = poses[i]
        images_names = sorted(os.listdir(video_path))
        n_frames = num_images

        matrix = np.linalg.inv(np.array(c2w))
        R = np.transpose(matrix[:3, :3])
        T = matrix[:3, 3]

        for idx, image_name in enumerate(images_names[:num_images]):
            image_path = os.path.join(video_path, image_name)
            image = Image.open(image_path)
            frame_time = idx / (n_frames - 1)

            FovX = focal2fov(focal, image.size[0])
            FovY = focal2fov(focal, image.size[1])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovX=FovX, FovY=FovY,
                                        image=image,
                                        image_path=image_path, image_name=image_name,
                                        width=image.size[0], height=image.size[1], fid=frame_time))

            idx += 1
    return cam_infos


def format_infos(dataset):
    # loading
    cameras = []
    for idx, (image, poses, time) in enumerate(tqdm(dataset, desc="Loading Neu3D")):
        image_path = dataset.image_paths[idx]
        image_name = '%04d.png' % idx
        # matrix = np.linalg.inv(np.array(pose))
        R, T = poses
        FovX = focal2fov(dataset.focal[0], image.size[0])
        FovY = focal2fov(dataset.focal[0], image.size[1])
        cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                            fid = time))

    return cameras

def readPlenopticVideoDataset(datadir, eval, num_images, hold_id=[0]):

    # loading all the data follow hexplane format
    ply_path = os.path.join(datadir, "points3D_downsample2.ply")
    pcd = fetchPly(ply_path)
    print("Find:", ply_path, "PCD:", pcd.points.shape)

    from scene.neu3d import Neural3D_NDC_Dataset
    train_dataset = Neural3D_NDC_Dataset(
    datadir,
    "train",
    1.0,
    time_scale=1,
    scene_bbox_min=[-2.5, -2.0, -1.0],
    scene_bbox_max=[2.5, 2.0, 1.0],
    eval_index=0,
        )    
    test_dataset = Neural3D_NDC_Dataset(
    datadir,
    "test",
    1.0,
    time_scale=1,
    scene_bbox_min=[-2.5, -2.0, -1.0],
    scene_bbox_max=[2.5, 2.0, 1.0],
    eval_index=0,
        )
    train_cam_infos = format_infos(train_dataset)
    test_cam_infos = format_infos(test_dataset)
    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,  # colmap dataset reader from official 3D Gaussian [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/]
    "Blender": readNerfSyntheticInfo,  # D-NeRF dataset [https://drive.google.com/file/d/1uHVyApwqugXTFuIRRlE4abTW8_rrVeIK/view?usp=sharing]
    "DTU": readNeuSDTUInfo,  # DTU dataset used in Tensor4D [https://github.com/DSaurus/Tensor4D]
    "nerfies": readNerfiesInfo,  # NeRFies & HyperNeRF dataset proposed by [https://github.com/google/hypernerf/releases/tag/v0.1]
    "plenopticVideo": readPlenopticVideoDataset,  # Neural 3D dataset in [https://github.com/facebookresearch/Neural_3D_Video]
}
