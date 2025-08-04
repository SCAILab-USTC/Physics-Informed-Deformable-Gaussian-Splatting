import torch
import torch.nn as nn
import torch.nn.functional as F
from scene_PIDG.network import DeformNetwork, Grid4D
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func


class DeformModel:
    def __init__(
            self, 
            grid_args,
            net_args,
            spatial_downsample_ratio=0.1,
            spatial_perturb_range=1e-3,
            temporal_downsample_ratio=0.1,
            temporal_perturb_range=1e-2,
            scale_xyz=1.0,
            reg_spatial_able=True,
            reg_temporal_able=True,
        ):
        self.grid4d = Grid4D(**grid_args).cuda()
        self.spatial_dim = self.grid4d.canonical_level_dim * self.grid4d.canonical_num_levels
        self.temporal_dim = self.grid4d.deform_level_dim * self.grid4d.deform_num_levels * 3
        self.deform = DeformNetwork(spatial_in_dim=self.spatial_dim, temporal_in_dim=self.temporal_dim, **net_args).cuda()

        self.optimizer = None
        self.network_lr_scale = 5.0
        self.grid_lr_scale = 100.0
        self.spatial_downsample_ratio = spatial_downsample_ratio
        self.temporal_downsample_ratio = temporal_downsample_ratio

        self.reg_spatial_able = reg_spatial_able
        self.spatial_perturb_range = None
        if self.reg_spatial_able:
            if type(spatial_perturb_range) is float:
                spatial_perturb_range = [spatial_perturb_range for _ in range(3)]
            else:
                assert len(spatial_perturb_range) == 3
            self.spatial_perturb_range = torch.tensor(spatial_perturb_range, device="cuda", dtype=torch.float32)

        self.reg_temporal_able = reg_temporal_able
        self.temporal_perturb_range = None
        if self.reg_temporal_able:
            if type(temporal_perturb_range) is float:
                temporal_perturb_range = [temporal_perturb_range for _ in range(4)]
            else:
                assert len(temporal_perturb_range) == 4
            self.temporal_perturb_range = torch.tensor(temporal_perturb_range, device="cuda", dtype=torch.float32)


        if type(scale_xyz) is float:
                scale_xyz = [scale_xyz for _ in range(3)]
        else:
            assert len(scale_xyz) == 3
        self.scale_xyz = torch.tensor(scale_xyz, device="cuda", dtype=torch.float32)
        
    def step(self, xyz, t, fixed_attention=False):
        xyz = xyz * self.scale_xyz[None, ...]
        t = (t * 2 * self.grid4d.bound - self.grid4d.bound) * 0.9
        xyzt = torch.cat([xyz, t], dim=-1)

        # get feature
        if fixed_attention and self.deform.attention_score is not None:
            temporal_h = self.grid4d.encode_temporal(xyzt)
            spatial_h = None
        else:
            spatial_h, temporal_h = self.grid4d(xyzt)
        d_xyz, rotation, scaling = self.deform(spatial_h, temporal_h, fixed_attention)

        # spatial regularization
        if self.reg_spatial_able:
            if self.spatial_downsample_ratio < 1.0:
                choice = torch.randperm(xyz.shape[0], device=xyz.device)[: int(max(1, xyz.shape[0] * self.spatial_downsample_ratio))]
                xyz = xyz[choice]
                spatial_h = spatial_h[choice]

            xyz_perturb = xyz + (torch.rand_like(xyz) * 2.0 - 1.0) * self.spatial_perturb_range[None, ...]
            spatial_h_perturb = self.grid4d.encode_spatial(xyz_perturb)
            reg_spatial = torch.sum(torch.abs(spatial_h_perturb - spatial_h) ** 2, dim=-1)
        else:
            reg_spatial = None

        # temporal regularization
        if self.reg_temporal_able:
            if self.temporal_downsample_ratio < 1.0:
                choice = torch.randperm(xyzt.shape[0], device=xyzt.device)[: int(max(1, xyzt.shape[0] * self.temporal_downsample_ratio))]
                xyzt = xyzt[choice]
                temporal_h = temporal_h[choice]

            xyzt_perturb = xyzt + (torch.rand_like(xyzt) * 2.0 - 1.0) * self.temporal_perturb_range[None, ...]
            temporal_h_perturb = self.grid4d.encode_temporal(xyzt_perturb)
            reg_temporal = torch.sum(torch.abs(temporal_h_perturb - temporal_h) ** 2, dim=-1)
        else:
            reg_temporal = None
            
        return {
            "d_xyz": d_xyz, 
            "d_rotation": rotation, 
            "d_scaling": scaling, 
            "reg_spatial": reg_spatial,
            "reg_temporal": reg_temporal,
        }
    
    def train_setting(self, training_args):
        self.network_lr_scale = training_args.network_lr_scale
        self.grid_lr_scale = training_args.grid_lr_scale

        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.position_lr_init * self.network_lr_scale,
             "name": "deform"},
            {'params': list(self.grid4d.parameters()),
             'lr': training_args.position_lr_init * self.grid_lr_scale,
             "name": "grid"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.network_lr_scheduler = get_expon_lr_func(lr_init=training_args.position_lr_init * self.network_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)
        self.grid_lr_scheduler = get_expon_lr_func(lr_init=training_args.position_lr_init * self.grid_lr_scale,
                                                       lr_final=training_args.position_lr_final * self.grid_lr_scale,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)

    def save_weights(self, model_path, iteration, is_best=False):
        if is_best:
            out_weights_path = os.path.join(model_path, "deform/iteration_best")
            os.makedirs(out_weights_path, exist_ok=True)
            with open(os.path.join(out_weights_path, "iter.txt"), "w") as f:
                f.write("Best iter: {}".format(iteration))
        else:
            out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
            os.makedirs(out_weights_path, exist_ok=True)
        torch.save((self.grid4d.state_dict(), self.deform.state_dict()), os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
            weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        else:
            loaded_iter = iteration
            weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))

        print("Load weight:", weights_path)
        grid_weight, network_weight = torch.load(weights_path, map_location='cuda')
        self.deform.load_state_dict(network_weight)
        self.grid4d.load_state_dict(grid_weight)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.network_lr_scheduler(iteration)
                param_group['lr'] = lr
            elif param_group['name'] == 'grid':
                lr = self.grid_lr_scheduler(iteration)
                param_group['lr'] = lr
