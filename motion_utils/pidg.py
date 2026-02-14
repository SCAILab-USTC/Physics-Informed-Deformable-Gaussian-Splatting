import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from math import exp

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
    def __init__(self, plane_resolution=32, feature_dim=4, enable_index_embedding=True, max_particles=100000):
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

def compute_acceleration(velocity_net, coords, dt):
    """计算加速度 ∂v/∂t
    方法：通过自动微分计算速度对时间的偏导数
    公式：dv/dt = ∂v/∂t + (v·∇)v （此处仅计算时间导数部分）
    """
    velocity, _ = velocity_net(coords)  # 获取速度场 [batch, 3]
    dvdt_list = []
    for i in range(3):
        # 计算每个速度分量的时间导数
        grad = torch.autograd.grad(
            outputs=velocity[:,i],
            inputs=coords,
            grad_outputs=torch.ones_like(velocity[:,i]),
            create_graph=True,
            retain_graph=True
        )[0]  # [batch, 4] (梯度包含空间和时间导数)
        dvdt_list.append(grad[:,3])  # 取时间导数分量（第4维）
    return torch.stack(dvdt_list, dim=1)  # [batch, 3]

def compute_stress_divergence(coords, velocity, sigma_params, force_sigma):
    """计算应力张量的散度 ∇·σ
    方法：
    1. 构建各向异性应力张量 σ
    2. 添加各向同性项：σ += μ(ε - 1/3 tr(ε)I)
    3. 计算散度分量：∑_j ∂σ_ij/∂x_j
    
    公式：
    ε = 0.5(∇v + (∇v)^T)  # 应变率张量
    σ = σ_anisotropic + μ(ε - 1/3 tr(ε)I)
    """
    # 计算速度梯度 ∇v [batch, 3, 3]
    grad_v = torch.stack([
        torch.autograd.grad(
            velocity[:,i], coords,
            grad_outputs=torch.ones_like(velocity[:,i]),
            create_graph=True,
            retain_graph=True
        )[0][:, :3] for i in range(3)  # 取空间导数（前3维）
    ], dim=1)
    
    # 计算应变率张量 ε = 0.5(∇v + ∇v^T)
    strain_rate = 0.5 * (grad_v + grad_v.transpose(1,2))  # [batch, 3, 3]
    
    # 构建各向异性应力张量 σ
    sigma = torch.zeros(coords.size(0), 3, 3, device=coords.device)
    sigma[:,0,0] = sigma_params[:,0]  # σ_xx
    sigma[:,1,1] = sigma_params[:,1]  # σ_yy
    sigma[:,2,2] = sigma_params[:,2]  # σ_zz
    sigma[:,0,1] = sigma[:,1,0] = sigma_params[:,3]  # σ_xy
    sigma[:,0,2] = sigma[:,2,0] = sigma_params[:,4]  # σ_xz
    sigma[:,1,2] = sigma[:,2,1] = sigma_params[:,5]  # σ_yz
    
    # 添加各向同性项：σ += μ(ε - (tr(ε)/3)I)
    trace_eps = strain_rate[:,0,0] + strain_rate[:,1,1] + strain_rate[:,2,2]  # [batch]
    identity = torch.eye(3, device=coords.device).unsqueeze(0).expand(coords.size(0), -1, -1)  # [batch,3,3]
    sigma += force_sigma * (strain_rate - (trace_eps.view(-1,1,1)/3) * identity)
    
    # 计算应力张量的散度 ∇·σ
    div_sigma = torch.stack([
        # 第i个分量的散度：∑_j ∂σ_ij/∂x_j 
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
    完整柯西动量方程：
      ρ (∂v/∂t + (v·∇)v) − ∇·σ = 0
    其中 v, σ 都由 velocity_net(coords) 给出。
    """
    # v: [N,3], σ_params: [N,6]
    velocity, sigma_params = velocity_net(coords)

    # 时间导数项 ∂v/∂t
    dvdt = compute_acceleration(velocity_net, coords, dt)  # [N,3]

    # 空间梯度 ∇v, 只取前三列
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

    # σ 的散度
    div_sigma = compute_stress_divergence(coords, velocity, sigma_params, force_sigma)  # [N,3]

    # 对流项 (v·∇)v
    convective = torch.einsum('bi,bij->bj', velocity, grad_v)  # [N,3]

    # 最终残差
    residual = rho * (dvdt + convective) - div_sigma  # [N,3]
    return torch.mean(residual**2)