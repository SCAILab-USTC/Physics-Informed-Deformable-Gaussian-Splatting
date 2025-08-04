from scene_PIDG.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch, ArrayToTorch
from utils.graphics_utils import fov2focal
import json
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor

WARNED = False


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    if (orig_w, orig_h) == resolution:
        resized_image_rgb = pil_to_tensor(cam_info.image).float().div_(255.0)
        flow_resized=cam_info.flow
        mask_resized=cam_info.mask_image
        depth_resized=cam_info.depth
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        mask_resized, depth_resized, flow_resized = None, None, None
        resolution = resolution[::-1]
        if cam_info.mask_image is not None:
            mask_resized = F.interpolate(torch.from_numpy(cam_info.mask_image).permute(2,0,1).unsqueeze(0).float(), size=resolution,mode="nearest")[0]        # [1,H,W]->[1,1,H,W]->[new_h,new_w]
            mask_resized = mask_resized.numpy() # [1,H,W]
        if cam_info.depth is not None:
            depth_resized = F.interpolate(cam_info.depth[None,None,...].float(),size=resolution, mode="bilinear", align_corners=False)[0,0]  # [new_h,new_w]
        flow_fwd, flow_bwd, _, _ = cam_info.flow or (None, None, None, None)
        flow_fwd_resized = flow_bwd_resized = None
        if flow_fwd is not None:
            flow_fwd_t = flow_fwd.permute(2,0,1).unsqueeze(0)
            flow_fwd_resized = F.interpolate(flow_fwd_t, size=resolution, mode="bilinear", align_corners=False)
            flow_fwd_resized = flow_fwd_resized[0].permute(1,2,0)           # [new_h,new_w,2]
            flow_fwd_resized[...,0] *= 1/scale
            flow_fwd_resized[...,1] *= 1/scale
            
        if flow_bwd is not None:
            flow_bwd_t = flow_bwd.permute(2,0,1).unsqueeze(0)
            flow_bwd_resized = F.interpolate(flow_bwd_t, size=resolution, mode="bilinear", align_corners=False)
            flow_bwd_resized = flow_bwd_resized[0].permute(1,2,0)           # [new_h,new_w,2]
            flow_bwd_resized[...,0] *= 1/scale
            flow_bwd_resized[...,1] *= 1/scale
        
        flow_resized = (flow_fwd_resized, flow_bwd_resized, None, None)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id,
                  data_device=args.data_device if not args.load2gpu_on_the_fly else 'cpu', fid=cam_info.fid,
                  flow=flow_resized,mask=mask_resized,depth=depth_resized)

def loadCam_train(args, id, cam_info, resolution_scale, next_cam, prev_cam):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
    if (orig_w, orig_h) == resolution:
        resized_image_rgb = pil_to_tensor(cam_info.image).float().div_(255.0)
        flow_resized=cam_info.flow
        mask_resized=cam_info.mask_image
        depth_resized=cam_info.depth
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        mask_resized, depth_resized, flow_resized = None, None, None
        resolution = resolution[::-1]
        if cam_info.mask_image is not None:
            mask_resized = F.interpolate(torch.from_numpy(cam_info.mask_image).permute(2,0,1).unsqueeze(0).float(), size=resolution,mode="nearest")[0]        # [1,new_h,new_w]
            mask_resized = mask_resized.numpy()
        if cam_info.depth is not None:
            depth_resized = F.interpolate(cam_info.depth[None,None,...].float(),size=resolution, mode="bilinear", align_corners=False)[0,0]  # [new_h,new_w]
        flow_fwd, flow_bwd, _, _ = cam_info.flow
        flow_fwd_resized = flow_bwd_resized = None
        if flow_fwd is not None:
            flow_fwd_t = flow_fwd.permute(2,0,1).unsqueeze(0)
            flow_fwd_resized = F.interpolate(flow_fwd_t, size=resolution, mode="bilinear", align_corners=False)
            flow_fwd_resized = flow_fwd_resized[0].permute(1,2,0)           # [new_h,new_w,2]
            flow_fwd_resized[...,0] *= 1/scale
            flow_fwd_resized[...,1] *= 1/scale
            
        if flow_bwd is not None:
            flow_bwd_t = flow_bwd.permute(2,0,1).unsqueeze(0)
            flow_bwd_resized = F.interpolate(flow_bwd_t, size=resolution, mode="bilinear", align_corners=False)
            flow_bwd_resized = flow_bwd_resized[0].permute(1,2,0)           # [new_h,new_w,2]
            flow_bwd_resized[...,0] *= 1/scale
            flow_bwd_resized[...,1] *= 1/scale
        
        flow_resized = (flow_fwd_resized, flow_bwd_resized, None, None)
    
    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None
    if resized_image_rgb.shape[0] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
        
    # resize flow, mask and depth
    
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id,
                  data_device=args.data_device if not args.load2gpu_on_the_fly else 'cpu', fid=cam_info.fid,
                  flow=flow_resized,mask=mask_resized,depth=depth_resized,
                  next_cam=next_cam,prev_cam=prev_cam,)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        if isinstance(c, dict):
            next_cam = None
            prev_cam = None
            if c["cam_next"] is not None:
                next_cam = loadCam(args, id, c["cam_next"], resolution_scale)
            if c["cam_prev"] is not None:
                prev_cam = loadCam(args, id, c["cam_prev"], resolution_scale)
            cam = loadCam_train(args, id, c["cam"], resolution_scale, next_cam, prev_cam)
            camera_list.append(cam)
        else:
            camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id': id,
        'img_name': camera.image_name,
        'width': camera.width,
        'height': camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy': fov2focal(camera.FovY, camera.height),
        'fx': fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def camera_nerfies_from_JSON(path, scale):
    """Loads a JSON camera into memory."""
    with open(path, 'r') as fp:
        camera_json = json.load(fp)

    # Fix old camera JSON.
    if 'tangential' in camera_json:
        camera_json['tangential_distortion'] = camera_json['tangential']

    return dict(
        orientation=np.array(camera_json['orientation']),
        position=np.array(camera_json['position']),
        focal_length=camera_json['focal_length'] * scale,
        principal_point=np.array(camera_json['principal_point']) * scale,
        skew=camera_json['skew'],
        pixel_aspect_ratio=camera_json['pixel_aspect_ratio'],
        radial_distortion=np.array(camera_json['radial_distortion']),
        tangential_distortion=np.array(camera_json['tangential_distortion']),
        image_size=np.array((int(round(camera_json['image_size'][0] * scale)),
                             int(round(camera_json['image_size'][1] * scale)))),
    )
