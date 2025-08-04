grid_args = dict(
    canonical_num_levels=16,
    canonical_level_dim=2,
    canonical_base_resolution=16,
    canonical_desired_resolution=2048,
    canonical_log2_hashmap_size=19,

    deform_num_levels=32,
    deform_level_dim=2,
    deform_base_resolution=[8, 8, 8],
    deform_desired_resolution=[32, 32, 16],
    deform_log2_hashmap_size=19,

    bound=1.6,
)

network_args = dict(
    depth=0,
    width=256,
    directional=True,
)

grid_lr_scale = 50.0
network_lr_scale = 5.0

lambda_spatial_tv = 0.5
spatial_downsample_ratio = 0.1
spatial_perturb_range = 1e-2

lambda_temporal_tv = 0.5
temporal_downsample_ratio = 0.1
temporal_perturb_range = [1e-2, 1e-2, 1e-2, 1e-3]

use_flow = True
save_image_dir = "/opt/data/private/projects/Dynamic-Reconstruction/Ours-Projects/Flow-Based-Module/PIDG_static_dynamic/outputs/vis/balls_reaction_v3"