import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from math import exp
import os
from utils.system_utils import searchForMaxIteration

class FourierTimeEmbed(nn.Module):
    def __init__(self, n_frequencies=8):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.frequencies = nn.Parameter(torch.randn(n_frequencies) * 2 * torch.pi)

    def forward(self, t):
        t = t.view(-1, 1)
        sin_features = torch.sin(self.frequencies * t)
        cos_features = torch.cos(self.frequencies * t)
        return torch.cat([sin_features, cos_features], dim=-1)

class VelocityNet(nn.Module):
    def __init__(self, plane_resolution=32, feature_dim=4, enable_index_embedding=True, max_particles=1_200_000):
        super().__init__()
        self.time_embed = FourierTimeEmbed(n_frequencies=8)
        self.planes = nn.Parameter(torch.randn(6, plane_resolution, plane_resolution, feature_dim) * 0.01)
        self.resolution = plane_resolution
        self.feature_dim = feature_dim
        self.time_proj = nn.Linear(16, 1)

        self.enable_index_embedding = enable_index_embedding
        if enable_index_embedding:
            self.index_embed = nn.Embedding(max_particles, 16)
            input_dim = 3 + 6 * feature_dim + 16
        else:
            input_dim = 3 + 6 * feature_dim

        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.velocity_head = nn.Linear(64, 3)
        self.stress_head = nn.Linear(64, 6)
        with torch.no_grad():
            self.velocity_head.bias.zero_()

    def fast_bilinear_interp(self, plane, x, y):
        batch_size = x.shape[0]
        plane_size = self.resolution
        x = (x.clamp(-1, 1) + 1) * (plane_size - 1) / 2
        y = (y.clamp(-1, 1) + 1) * (plane_size - 1) / 2

        x0 = torch.floor(x).long().clamp(0, plane_size - 2)
        y0 = torch.floor(y).long().clamp(0, plane_size - 2)
        x1 = (x0 + 1).clamp(0, plane_size - 1)
        y1 = (y0 + 1).clamp(0, plane_size - 1)

        wx = (x - x0.float()).view(-1, 1)
        wy = (y - y0.float()).view(-1, 1)

        v00 = plane[x0, y0]
        v01 = plane[x0, y1]
        v10 = plane[x1, y0]
        v11 = plane[x1, y1]

        v0 = v00 * (1 - wx) + v10 * wx
        v1 = v01 * (1 - wx) + v11 * wx
        v = v0 * (1 - wy) + v1 * wy

        return v

    def forward(self, x, indices=None):
        xyz, t = x[:, :3], x[:, 3:]
        t_feat = self.time_embed(t)
        t_normalized = torch.tanh(self.time_proj(t_feat)).squeeze(-1)

        x_coord, y_coord, z_coord = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        features = [
            self.fast_bilinear_interp(self.planes[0], x_coord, y_coord),
            self.fast_bilinear_interp(self.planes[1], x_coord, z_coord),
            self.fast_bilinear_interp(self.planes[2], y_coord, z_coord),
            self.fast_bilinear_interp(self.planes[3], x_coord, t_normalized),
            self.fast_bilinear_interp(self.planes[4], y_coord, t_normalized),
            self.fast_bilinear_interp(self.planes[5], z_coord, t_normalized),
        ]

        plane_features = torch.cat(features, dim=1)
        combined_features = torch.cat([xyz, plane_features], dim=1)

        if self.enable_index_embedding and indices is not None:
            index_features = self.index_embed(indices)
            combined_features = torch.cat([combined_features, index_features], dim=1)

        shared_features = self.shared_net(combined_features)
        velocity = self.velocity_head(shared_features)
        stress_params = self.stress_head(shared_features)
        return velocity, stress_params
    
    def save_weights(self, model_path, iteration, is_best=False):
        if is_best:
            out_weights_path = os.path.join(model_path, "velocity/iteration_best")
            os.makedirs(out_weights_path, exist_ok=True)
            with open(os.path.join(out_weights_path, "iter.txt"), "w") as f:
                f.write("Best iter: {}".format(iteration))
        else:
            out_weights_path = os.path.join(model_path, "velocity/iteration_{}".format(iteration))
            os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(out_weights_path, 'velocity_net.pth'))
        
    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "velocity"))
            weights_path = os.path.join(model_path, "velocity/iteration_{}/velocity_net.pth".format(loaded_iter))
        else:
            loaded_iter = iteration
            weights_path = os.path.join(model_path, "velocity/iteration_{}/velocity_net.pth".format(loaded_iter))

        print("Load weight:", weights_path)
        velocity_state = torch.load(weights_path, map_location='cuda')
        self.load_state_dict(velocity_state)

def compute_acceleration(velocity_net, coords, dt):
    """
    Compute the acceleration ∂v/∂t.
    Method: use automatic differentiation to get the time derivative of velocity.
    Formula: dv/dt = ∂v/∂t + (v·∇)v (only the time derivative part is computed here).
    """
    velocity, _ = velocity_net(coords)  # get velocity field [batch, 3]
    dvdt_list = []
    for i in range(3):
        # compute the time derivative of each velocity component
        grad = torch.autograd.grad(
            outputs=velocity[:,i],
            inputs=coords,
            grad_outputs=torch.ones_like(velocity[:,i]),
            create_graph=True,
            retain_graph=True
        )[0]  # [batch, 4] (gradient contains spatial and temporal derivatives)
        dvdt_list.append(grad[:,3])  # take the time derivative component (4th dim)
    return torch.stack(dvdt_list, dim=1)  # [batch, 3]

def compute_stress_divergence(coords, velocity, sigma_params, force_sigma):
    """
    Compute the divergence of the stress tensor ∇·σ.
    Method:
    1. Construct the anisotropic stress tensor σ
    2. Add the isotropic correction: σ += μ(ε - 1/3 tr(ε)I)
    3. Compute divergence components: ∑_j ∂σ_ij/∂x_j
    
    Formulas:
    ε = 0.5(∇v + (∇v)^T)  # strain rate tensor
    σ = σ_anisotropic + μ(ε - 1/3 tr(ε)I)
    """
    # compute velocity gradient ∇v [batch, 3, 3]
    grad_v = torch.stack([
        torch.autograd.grad(
            velocity[:,i], coords,
            grad_outputs=torch.ones_like(velocity[:,i]),
            create_graph=True,
            retain_graph=True
        )[0][:, :3] for i in range(3)  # take spatial derivatives (first 3 dims)
    ], dim=1)
    
    # compute strain rate tensor ε = 0.5(∇v + ∇v^T)
    strain_rate = 0.5 * (grad_v + grad_v.transpose(1,2))  # [batch, 3, 3]
    
    # build anisotropic stress tensor σ
    sigma = torch.zeros(coords.size(0), 3, 3, device=coords.device)
    sigma[:,0,0] = sigma_params[:,0]  # σ_xx
    sigma[:,1,1] = sigma_params[:,1]  # σ_yy
    sigma[:,2,2] = sigma_params[:,2]  # σ_zz
    sigma[:,0,1] = sigma[:,1,0] = sigma_params[:,3]  # σ_xy
    sigma[:,0,2] = sigma[:,2,0] = sigma_params[:,4]  # σ_xz
    sigma[:,1,2] = sigma[:,2,1] = sigma_params[:,5]  # σ_yz
    
    # add isotropic term: σ += μ(ε - (tr(ε)/3)I)
    trace_eps = strain_rate[:,0,0] + strain_rate[:,1,1] + strain_rate[:,2,2]  # [batch]
    identity = torch.eye(3, device=coords.device).unsqueeze(0).expand(coords.size(0), -1, -1)  # [batch,3,3]
    sigma += force_sigma * (strain_rate - (trace_eps.view(-1,1,1)/3) * identity)
    
    # compute divergence of stress tensor ∇·σ
    div_sigma = torch.stack([
        # divergence of the i-th component: ∑_j ∂σ_ij/∂x_j 
        torch.autograd.grad(
            sigma[:,i,0], coords,
            grad_outputs=torch.ones_like(sigma[:,i,0]),
            create_graph=True
        )[0][:,0] +  # ∂σ_ix/∂x
        torch.autograd.grad(
            sigma[:,i,1], coords,
            grad_outputs=torch.ones_like(sigma[:,i,1]),
            create_graph=True
        )[0][:,1] +  # ∂σ_iy/∂y
        torch.autograd.grad(
            sigma[:,i,2], coords,
            grad_outputs=torch.ones_like(sigma[:,i,2]),
            create_graph=True
        )[0][:,2]    # ∂σ_iz/∂z
        for i in range(3)
    ], dim=1)  # [batch, 3]
    
    return div_sigma

def cauchy_momentum_constraint(coords, velocity_net, force_sigma, dt=1e-3, rho=1.0):
    """
    Full Cauchy momentum equation:
      ρ (∂v/∂t + (v·∇)v) − ∇·σ = 0
    Where v and σ are produced by velocity_net(coords).
    """
    # v: [N,3], σ_params: [N,6]
    velocity, sigma_params = velocity_net(coords)

    # time derivative term ∂v/∂t
    dvdt = compute_acceleration(velocity_net, coords, dt)  # [N,3]

    # spatial gradient ∇v, taking the first three spatial dimensions
    grad_v = torch.stack([
        torch.autograd.grad(
            outputs=velocity[:, i],
            inputs=coords,
            grad_outputs=torch.ones_like(velocity[:, i]),
            create_graph=True,
            retain_graph=True
        )[0][:, :3]
        for i in range(3)
    ], dim=1)  # [N,3,3]

    # divergence of σ
    div_sigma = compute_stress_divergence(coords, velocity, sigma_params, force_sigma)  # [N,3]

    # convective term (v·∇)v
    convective = torch.einsum('bi,bij->bj', velocity, grad_v)  # [N,3]

    # final residual
    residual = rho * (dvdt + convective) - div_sigma  # [N,3]
    return torch.mean(residual**2)