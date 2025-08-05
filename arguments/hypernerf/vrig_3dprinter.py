grid_args = dict(
    canonical_num_levels=16,
    canonical_level_dim=2,
    canonical_base_resolution=16,
    canonical_desired_resolution=256,#2048,
    canonical_log2_hashmap_size=19,

    deform_num_levels=32,
    deform_level_dim=2,
    deform_base_resolution=[16, 16, 8],
    deform_desired_resolution=[512, 512, 32],
    deform_log2_hashmap_size=19,

    bound=1.0,
)

network_args = dict(
    depth=1,
    width=256,
    directional=True,
)

grid_lr_scale = 50.0
network_lr_scale = 1.0

lambda_spatial_tv = 1.0
spatial_downsample_ratio = 1.0
spatial_perturb_range = 1e-2

lambda_temporal_tv = 1.0
temporal_downsample_ratio = 1.0
temporal_perturb_range = [1e-2, 1e-2, 1e-2, 0.0]

warm_up = 1000
iterations = 40_000
deform_lr_max_steps = 40_000
position_lr_max_steps = 35_000

use_flow = True
dataset_type = "Hyper"