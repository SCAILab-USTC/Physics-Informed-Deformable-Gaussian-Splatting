import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from hashencoder.hashgrid import HashEncoder

def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class Grid4D(nn.Module):
    def __init__(
            self,
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
        ):
        super(Grid4D, self).__init__()
        self.out_dim = canonical_num_levels * canonical_level_dim + deform_num_levels * deform_level_dim * 3
        self.canonical_num_levels = canonical_num_levels
        self.canonical_level_dim = canonical_level_dim
        self.deform_num_levels = deform_num_levels
        self.deform_level_dim = deform_level_dim
        self.bound = bound

        self.xyz_encoding = HashEncoder(
            input_dim=3,
            num_levels=canonical_num_levels,
            level_dim=canonical_level_dim, 
            per_level_scale=2,
            base_resolution=canonical_base_resolution, 
            log2_hashmap_size=canonical_log2_hashmap_size,
            desired_resolution=canonical_desired_resolution,
        )

        self.xyt_encoding = HashEncoder(
            input_dim=3, 
            num_levels=deform_num_levels, 
            level_dim=deform_level_dim,
            per_level_scale=2,
            base_resolution=deform_base_resolution,
            log2_hashmap_size=deform_log2_hashmap_size,
            desired_resolution=deform_desired_resolution,
        )

        self.yzt_encoding = HashEncoder(
            input_dim=3, 
            num_levels=deform_num_levels, 
            level_dim=deform_level_dim,
            per_level_scale=2,
            base_resolution=deform_base_resolution,
            log2_hashmap_size=deform_log2_hashmap_size,
            desired_resolution=deform_desired_resolution,
        )

        self.xzt_encoding = HashEncoder(
            input_dim=3, 
            num_levels=deform_num_levels, 
            level_dim=deform_level_dim,
            per_level_scale=2,
            base_resolution=deform_base_resolution,
            log2_hashmap_size=deform_log2_hashmap_size,
            desired_resolution=deform_desired_resolution,
        )
    
    def encode_spatial(self, xyz):
        return self.xyz_encoding(xyz, size=self.bound)
    
    def encode_temporal(self, xyzt):
        xyt = torch.cat([xyzt[..., :2], xyzt[..., 3:]], dim=-1)
        yzt = xyzt[..., 1:]
        xzt = torch.cat([xyzt[..., :1], xyzt[..., 2:]], dim=-1)
        h = torch.cat([
            self.xyt_encoding(xyt, size=self.bound),
            self.yzt_encoding(yzt, size=self.bound),
            self.xzt_encoding(xzt, size=self.bound),
        ], dim=-1)

        return h

    def forward(self, xyzt):
        xyz = xyzt[..., :3]
        return self.encode_spatial(xyz), self.encode_temporal(xyzt)


class DeformNetwork(nn.Module):
    def __init__(
            self,
            spatial_in_dim,
            temporal_in_dim,
            depth=1,
            width=256,
            directional=True,
        ):
        super(DeformNetwork, self).__init__()
        self.depth = depth
        self.width = width
        self.directional = directional

        self.spatial_mlp = nn.Sequential(
            nn.Linear(spatial_in_dim, width),
        )

        self.temporal_mlp = nn.Sequential(
            nn.Linear(temporal_in_dim, width),
            nn.ReLU(),
        )

        mlp = []
        for _ in range(depth):
            mlp.append(nn.Linear(width, width))
            mlp.append(nn.ReLU())
        self.grid_mlp = nn.Sequential(*mlp)

        self.gaussian_warp = nn.Linear(width, 7)
        self.gaussian_rotation = nn.Linear(width, 4)
        self.gaussian_scaling = nn.Linear(width, 3)

        self.quat_bias = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

        self.attention_score = None

    def forward(self, spatial_h, temporal_h, fixed_attention=False):
        if fixed_attention and self.attention_score is not None:
            spatial_h = self.attention_score
        else:
            spatial_h = self.spatial_mlp(spatial_h)
            spatial_h = torch.sigmoid(spatial_h)
            if self.directional:
                spatial_h = spatial_h * 2.0 - 1.0
            if fixed_attention:
                self.attention_score = spatial_h
                
        h = self.temporal_mlp(temporal_h) * spatial_h

        h = self.grid_mlp(h)

        d_xyz = self.gaussian_warp(h) + self.quat_bias
        scaling = self.gaussian_scaling(h)
        rotation = self.gaussian_rotation(h)

        return d_xyz, rotation, scaling
