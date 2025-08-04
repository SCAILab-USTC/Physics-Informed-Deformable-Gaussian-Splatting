grid_args = dict(
    canonical_num_levels=16,
    canonical_level_dim=2,
    canonical_base_resolution=16,
    canonical_desired_resolution=2048,
    canonical_log2_hashmap_size=19,

    deform_num_levels=32,
    deform_level_dim=2,
    deform_base_resolution=[16, 16, 8],
    deform_desired_resolution=[512, 512, 32],
    deform_log2_hashmap_size=19,

    bound=1.6,
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
spatial_perturb_range = 1e-3

lambda_temporal_tv = 1.0
temporal_downsample_ratio = 1.0
temporal_perturb_range = [1e-2, 1e-2, 1e-2, 0.0]

warm_up = 1_000
opacity_reset_interval = 1000
densify_until_iter = 10_000
iterations = 40_000
deform_lr_max_steps = 25_000
position_lr_max_steps = 25_000

use_flow = True
save_image_dir = "/mnt/store2/fd/project/3D-Reconstruction/Dynamic/PIDG_static_dynamic/outputs/vis/chicken"