# ------------------------------------------------------------------------
# Modified from Conditional DETR Transformer class.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import copy
import math
from typing import Optional, List
import argparse
import random 
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from omegaconf import OmegaConf, DictConfig

from .ms_deform_attn import MSDeformAttn
from .dn_component import prepare_for_cdn, dn_post_process
from .utils import gen_encoder_output_proposals, inverse_sigmoid, MLP, _get_activation_fn

from ...misc.keypoint_ops import keypoint_xyzxyz_to_xyxyzz

# from kan import KAN 

import torch
import torch.nn as nn
import torch.nn.functional as F

class FastKANLayer(nn.Module):
    """
    Fast KAN layer (RBF approximation of KAN edge-splines).

    Maps x (.., in_features) -> y (.., out_features).

    Forward math (intuitively):
        u_p = scaled_univariate(x_p)                       # per-dim scalar mapped to grid coordinate
        phi_{q,p}(x_p) = sum_k c[q,p,k] * RBF_k(u_p)       # RBF basis on grid
        s_q = sum_p phi_{q,p}(x_p)                         # inner sums (r inner functions)
        y = Linear(s)                                      # linear outer mixing to get out_features

    Parameters:
        in_features (int): input dimensionality (D)
        out_features (int): output dimensionality (M)
        r (int, optional): number of inner functions (default: 2*D + 1, as in Kolmogorov constructions)
        grid_size (int): number of RBF centers per 1D function (default 21)
        sigma (float): RBF bandwidth (default 1.0). Smaller -> sharper RBFs.
        use_layernorm (bool): whether to apply LayerNorm to inputs before coordinate mapping.
        init_scale (float): initialization scale for coefficients.

    Shapes:
        coeffs: (r, D, grid_size)
        basis (computed): (...batch..., D, grid_size)
        s: (...batch..., r)
        output: (...batch..., out_features)
    """
    def __init__(self, in_features, out_features,
                 r=None, grid_size: int = 21, sigma: float = 1.0,
                 use_layernorm: bool = True, init_scale: float = 1e-2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r if (r is not None) else (in_features//2 + 1)
        self.grid_size = int(grid_size)
        self.sigma = float(sigma)
        self.use_layernorm = use_layernorm

        if use_layernorm:
            self.ln = nn.LayerNorm(in_features, eps=1e-6)
        else:
            self.ln = None

        # coefficients for inner univariate functions (one spline/RBF weight vector per (q, p))
        # shape: (r, D, G)
        self.coeffs = nn.Parameter(torch.randn(self.r, in_features, self.grid_size) * init_scale)

        # outer linear mixing: s (r) -> out_features
        self.out_linear = nn.Linear(self.r, out_features)

        # grid centers (0..G-1) stored as buffer (device moved automatically with module)
        centers = torch.arange(self.grid_size, dtype=torch.float32)
        self.register_buffer('centers', centers)

    def _map_to_grid(self, x):
        """
        Map normalized x to grid coordinates in [0, grid_size-1].
        Uses tanh to robustly squash LayerNorm outputs into [-1,1], then maps to [0, G-1].
        x: (..., D)
        returns: (..., D) with values in [0, G-1]
        """
        # x assumed already normalized by LayerNorm if used
        # use tanh so outliers get pushed inwards
        scaled = (torch.tanh(x) + 1.0) * 0.5  # now in [0,1]
        u = scaled * (self.grid_size - 1)
        return u

    def forward(self, x):
        """
        x: Tensor of shape (B, D) or (..., D). Last dim must be in_features.
        returns: Tensor with shape (..., out_features).
        """
        if x.shape[-1] != self.in_features:
            raise ValueError(f'Last dimension of input must be {self.in_features}, got {x.shape[-1]}')

        orig_shape = x.shape[:-1]  # may be batch or batch+sequence dims
        Bflat = int(torch.tensor(orig_shape).prod().item()) if len(orig_shape) > 0 else 1
        # flatten leading dims to single batch for easier computation
        x_flat = x.reshape(-1, self.in_features)  # (Bflat, D)

        if self.ln is not None:
            xn = self.ln(x_flat)
        else:
            xn = x_flat

        # map to grid coordinates u in [0, G-1]
        u = self._map_to_grid(xn)   # (Bflat, D)


        # compute Gaussian RBF basis: basis[b, p, k] = exp(-( (u[b,p] - centers[k]) / sigma)^2 )
        # centers shape (G,)
        # resulting basis shape: (Bflat, D, G)
        diff = u.unsqueeze(-1) - self.centers.view(1, 1, -1)  # (Bflat, D, G)
        basis = torch.exp(- (diff / (self.sigma + 1e-12)) ** 2)

        # compute s: s[b, q] = sum_p sum_k basis[b,p,k] * coeffs[q,p,k]
        # coeffs shape: (r, D, G)
        s = torch.einsum('bdg,rdg->br', basis, self.coeffs)  # (Bflat, r)

        # optional nonlinearity could be applied to s here (we keep linear outer mixing for simplicity)
        out = self.out_linear(s)  # (Bflat, out_features)

        # reshape back to original leading shape
        final_shape = tuple(orig_shape) + (self.out_features,)
        out = out.view(final_shape)
        return out

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, r={self.r}, "
                f"grid_size={self.grid_size}, sigma={self.sigma}, use_layernorm={self.use_layernorm}")



def _get_clones(module, N, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def weighting_function(reg_max, up, reg_scale, deploy=False):
    """
    Generates the non-uniform Weighting Function W(n) for bounding box regression.

    Args:
        reg_max (int): Max number of the discrete bins.
        up (Tensor): Controls upper bounds of the sequence,
                     where maximum offset is ±up * H / W.
        reg_scale (float): Controls the curvature of the Weighting Function.
                           Larger values result in flatter weights near the central axis W(reg_max/2)=0
                           and steeper weights at both ends.
        deploy (bool): If True, uses deployment mode settings.

    Returns:
        Tensor: Sequence of Weighting Function.
    """
    if deploy:
        upper_bound1 = (abs(up[0]) * abs(reg_scale)).item()
        upper_bound2 = (abs(up[0]) * abs(reg_scale) * 2).item()
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = (
            [-upper_bound2]
            + left_values
            + [torch.zeros_like(up[0][None])]
            + right_values
            + [upper_bound2]
        )
        return torch.tensor([values], dtype=up.dtype, device=up.device)
    else:
        upper_bound1 = abs(up[0]) * abs(reg_scale)
        upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = (
            [-upper_bound2]
            + left_values
            + [torch.zeros_like(up[0][None])]
            + right_values
            + [upper_bound2]
        )
        return torch.cat(values, 0)


def distance2pose(points, distance, reg_scale):
    """
    Decodes edge-distances into bounding box coordinates.

    Args:
        points (Tensor): (B, N, 4) or (N, 4) format, representing [x, y, w, h],
                         where (x, y) is the center and (w, h) are width and height.
        distance (Tensor): (B, N, 4) or (N, 4), representing distances from the
                           point to the left, top, right, and bottom boundaries.

        reg_scale (float): Controls the curvature of the Weighting Function.

    Returns:
        Tensor: Bounding boxes in (N, 4) or (B, N, 4) format [cx, cy, w, h].
    """
    reg_scale = abs(reg_scale)
    x1 = points[..., 0] + distance[..., 0] / reg_scale
    y1 = points[..., 1] + distance[..., 1] / reg_scale

    pose = torch.stack([x1, y1], -1)

    return pose


class Gate(nn.Module):
    def __init__(self, d_model):
        super(Gate, self).__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = float(-math.log((1 - 0.5) / 0.5))
        nn.init.constant_(self.gate.bias, bias)
        nn.init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = F.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)


class Integral(nn.Module):
    """
    A static layer that calculates integral results from a distribution.

    This layer computes the target location using the formula: `sum{Pr(n) * W(n)}`,
    where Pr(n) is the softmax probability vector representing the discrete
    distribution, and W(n) is the non-uniform Weighting Function.

    Args:
        reg_max (int): Max number of the discrete bins. Default is 32.
                       It can be adjusted based on the dataset or task requirements.
    """

    def __init__(self, reg_max=32):
        super(Integral, self).__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])


class LQE(nn.Module):
    def __init__(self, topk, hidden_dim, num_layers, num_body_points):
        super().__init__()
        self.k = topk
        self.hidden_dim = hidden_dim
        self.reg_conf = MLP(num_body_points * (topk+1), hidden_dim, 1, num_layers)
        nn.init.constant_(self.reg_conf.layers[-1].weight.data, 0)
        nn.init.constant_(self.reg_conf.layers[-1].bias.data, 0)

        self.num_body_points = num_body_points

    def forward(self, scores, pred_poses, feat):
        B, L = pred_poses.shape[:2]
        pred_poses = pred_poses.reshape(B, L, self.num_body_points, 2)

        sampling_values = F.grid_sample(feat
                                        , 2*pred_poses-1, 
            mode='bilinear', padding_mode='zeros', align_corners=False).permute(0, 2, 3, 1)
        
        prob_topk = sampling_values.topk(self.k, dim=-1)[0]

        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        # quality_score.register_hook(lambda x: print("quality score:",x.shape, x.isinf().sum(), (x==0).sum()))
        # scores.register_hook(lambda x: print("scores:",x.shape, x.isinf().sum(), (x==0).sum()))
        return scores + quality_score 


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, use_kan=False, kan_grid=3,
                 use_modulation=False, use_region_sampling=False, region_kernel_size=1,
                 use_global_context=False, use_grouped_offsets=False, num_groups=1,
                 use_grid_attention=False, grid_num_points=16, use_grid_offsets=False,
                 use_grid_fusion=True, is_energy=False, energy_in_dim=68, energy_out_dim=1):
        super().__init__()
        # within-instance self-attention
        self.within_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.within_dropout = nn.Dropout(dropout)
        self.within_norm = nn.LayerNorm(d_model)
        # across-instance self-attention
        self.across_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.across_dropout = nn.Dropout(dropout)
        self.across_norm = nn.LayerNorm(d_model)
        # deformable cross-attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, use_modulation=use_modulation, 
                                        use_region_sampling=use_region_sampling, region_kernel_size=region_kernel_size,
                                        use_global_context=use_global_context, use_grouped_offsets=use_grouped_offsets, num_groups=num_groups,
                                        use_grid_attention=use_grid_attention, grid_num_points=grid_num_points, use_grid_offsets=use_grid_offsets, use_grid_fusion=use_grid_fusion)
        self.dropout1 = nn.Dropout(dropout)
        # self.norm1 = nn.LayerNorm(d_model)
        # gate
        self.gateway = Gate(d_model)
        # FFN
        self.use_kan = use_kan
        self.is_energy = is_energy  # if True, this layer replaces EnergyHead
        self.energy_out_dim = energy_out_dim     # match the original EnergyHead output

        if self.is_energy:
            self.energy_expand = nn.Linear(energy_in_dim, d_model)
            self.energy_reduce = nn.Linear(d_model, self.energy_out_dim)
    


        # FFN: MLP or KAN
        if self.use_kan:
            # Two-layer KAN FFN
            self.kan1 = FastKANLayer(d_model, d_ffn, grid_size=kan_grid).to('cuda')
            self.kan2 = FastKANLayer(d_ffn, d_ffn, grid_size=kan_grid).to('cuda')
            self.out_linear = nn.Linear(d_ffn, d_model)
            self.activation = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout)
            self.dropout3 = nn.Dropout(dropout)
            self.norm2 = nn.LayerNorm(d_model)
        else:
            # Standard MLP FFN
            self.linear1 = nn.Linear(d_model, d_ffn)
            self.linear2 = nn.Linear(d_ffn, d_model)
            self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
            self.dropout2 = nn.Dropout(dropout)
            self.dropout3 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        if not self.use_kan:
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos, training):
        if pos is not None:
            np = pos.shape[2]
            # if training:
            #     x1, x2 = tensor.split([1, np], dim=2)
            #     x2 = x2 + pos
            #     tensor = torch.concat((x1, x2), dim=2)
            # else:
            #     tensor[:, :, -np:] += pos
            tensor[:, :, -np:] += pos
        return tensor
    def forward_FFN(self, tgt):
        if self.use_kan:
            # FastKAN forward
            bs, nq, num_kpt, d_model = tgt.shape
            t_flat = tgt.view(-1, d_model)
            out = self.dropout2(self.activation(self.kan1(t_flat)))
            out = self.dropout2(self.kan2(out))
            out = self.dropout3(self.out_linear(out))
            tgt = tgt + out.view(bs, nq, num_kpt, d_model)
            tgt = self.norm2(tgt)
        else:
            tgt2 = self.linear2(self.dropout2(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm2(tgt)
        return tgt

    def forward(self, 
                # for tgt
                tgt_pose: Optional[Tensor],
                tgt_pose_query_pos: Optional[Tensor],
                tgt_pose_reference_points: Optional[Tensor],
                attn_mask: Optional[Tensor] = None,
                # for memory
                memory: Optional[Tensor] = None,
                memory_spatial_shapes: Optional[Tensor] = None,
            ):
        if self.is_energy:
            tgt_pose = self.energy_expand(tgt_pose)

        bs, nq, num_kpt, d_model = tgt_pose.shape

        # within-instance self-attention
        q = k = self.with_pos_embed(tgt_pose, tgt_pose_query_pos, self.training).flatten(0, 1) # bs * nq, num_kpts, 2
        tgt2 = self.within_attn(q, k, tgt_pose.flatten(0, 1))[0].reshape(bs, nq, num_kpt, d_model)
        tgt_pose = tgt_pose + self.within_dropout(tgt2)
        tgt_pose = self.within_norm(tgt_pose)

        # across-instance self-attention
        tgt_pose = tgt_pose.transpose(1, 2).flatten(0, 1) # bs * num_kpts, nq, 2
        
        q_pose = k_pose = tgt_pose
        tgt2_pose = self.across_attn(
            q_pose, 
            k_pose, 
            tgt_pose,
            attn_mask=attn_mask
            )[0].reshape(bs * num_kpt, nq, d_model)
        tgt_pose = tgt_pose + self.across_dropout(tgt2_pose)
        tgt_pose = self.across_norm(tgt_pose).reshape(bs, num_kpt, nq, d_model).transpose(1, 2) # bs, nq, num_kpts, 2
            

        if not self.is_energy:
            # deformable cross-attention
            tgt2_pose = self.cross_attn(self.with_pos_embed(tgt_pose, tgt_pose_query_pos, self.training).flatten(1, 2),
                                            tgt_pose_reference_points,
                                            memory, #.transpose(0, 1), 
                                            memory_spatial_shapes, 
                                            ).reshape(bs, nq, num_kpt, d_model)

    
            tgt_pose = self.gateway(tgt_pose, self.dropout1(tgt2_pose))
            
        tgt_pose = self.forward_FFN(tgt_pose)
        if self.is_energy:
            # tgt_pose shape: (bs, nq, num_kpt, d_model)
            E = self.energy_reduce(tgt_pose)  # -> (bs, nq, num_kpt, 1)
            return E  # replace EnergyHead output
        else:
            return tgt_pose



# Modified TransformerDecoder with minimal, opt-in energy-based refinement
# (Everything else in the original loop/logic is preserved; only a few lines
#  were added to support an optional EnergyHead and inner-loop refinement.)

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Minimal EnergyHead used for the refinement (very small MLP)
class EnergyHead(nn.Module):
    """
    Minimal energy function E_theta(feat, z) -> scalar per instance.

    - feat: (bs, nq, C) conditioning features (we will use the per-query instance feature)
    - z_flat: (bs, nq, P) flattened keypoints (P = num_body_points * 2)
    returns:
    - energy: (bs, nq, 1)
    """
    def __init__(self, feat_dim: int, pose_dim: int, hidden: int = 256, n_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = feat_dim + pose_dim
        if n_layers <= 1:
            layers.append(nn.Linear(in_dim, 1))
        else:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden, hidden))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, feat: Tensor, z_flat: Tensor) -> Tensor:
        # feat: (bs, nq, C), z_flat: (bs, nq, P)
        x = torch.cat([feat, z_flat], dim=-1)
        bs, nq, _ = x.shape
        x = x.view(bs * nq, -1)
        e = self.net(x)
        e = e.view(bs, nq, 1)
        return e


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, 
                    return_intermediate=False, 
                    hidden_dim=256,
                    num_body_points=17,
                    # ---- new minimal energy options (kept optional and off by default) ----
                    use_energy_refinement = False,
                    energy_steps = 3,
                    energy_step_size = 0.1,
                    energy_hidden = 256,
                    energy_n_layers = 2,
                    energy_layer = None,
                    noise_scale = 0.01,
                    loss_all_steps=False,
                    energy_decrease_weight=0.0,
                    ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers, layer_share=False)
        else:
            self.layers = []
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_body_points = num_body_points
        self.return_intermediate = return_intermediate 
        self.class_embed = None
        self.pose_embed = None
        self.half_pose_ref_point_head = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        self.eval_idx = num_layers - 1
        self.noise_scale = noise_scale
        self.loss_all_steps = loss_all_steps
        self.energy_decrease_weight = energy_decrease_weight

        # for sin embedding
        dim_t = torch.arange(hidden_dim // 2, dtype=torch.float32)
        dim_t = 10000 ** (2 * (dim_t // 2) / (hidden_dim//2))
        self.register_buffer('dim_t', dim_t)
        self.scale = 2 * math.pi
        self.layer_loss = torch.zeros(1, dtype=torch.float32)

        # -------------------- energy refinement attributes (minimal) --------------------
        self.use_energy_refinement = use_energy_refinement
        if self.use_energy_refinement:
            print(">>> Using energy-based pose refinement")
            # step count and a learnable step size
            self.energy_steps = energy_steps
            self.energy_step_size = nn.Parameter(torch.tensor([energy_step_size], dtype=torch.float32))
            # energy head conditions on per-query instance features (hidden_dim) + flattened pose
            pose_dim = self.num_body_points * 2
            self.energy_head = energy_layer
     
        # -------------------------------------------------------------------------------

    def sine_embedding(self, pos_tensor):
        x_embed = pos_tensor[..., 0:1] * self.scale
        y_embed = pos_tensor[..., 1:2] * self.scale
        pos_x = x_embed / self.dim_t
        pos_y = y_embed / self.dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)
        
        if pos_tensor.size(-1) == 2:
            pos = torch.cat((pos_y, pos_x), dim=3)
        elif pos_tensor.size(-1) == 4:
            w_embed = pos_tensor[..., 2:3] * self.scale
            pos_w = w_embed / self.dim_t
            pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()), dim=4).flatten(3)

            h_embed = pos_tensor[..., 3:4] * self.scale
            pos_h = h_embed / self.dim_t
            pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()), dim=4).flatten(3)

            pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=3)
        else:
            raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
        return pos

    def _resolve_energy_steps(self, is_training=False):
        s = self.energy_steps
        if isinstance(s, int):
            return s
        if isinstance(s, dict):
            # random.choices accepts weights (doesn't need normalization)
            values = list(s.keys())
            weights = list(s.values())
            if is_training:
                return random.choices(values, weights=weights, k=1)[0]
            else:
                return values[weights.index(max(weights))]

        if callable(s):
            return int(s())
        raise TypeError("energy_steps must be int, dict, or callable")
    def forward(self, tgt, memory,
                refpoints_sigmoid,
                # prediction heads
                pre_pose_head,
                pose_head,
                class_head,
                lqe_head,
                # feature map
                feat_lqe,
                # new arguments
                integral,
                up,
                reg_scale,
                reg_max,
                project,
                # attention 
                attn_mask=None,
                # for memory
                spatial_shapes: Optional[Tensor] = None,
                ):
        
        output = tgt
        refpoint_pose = refpoints_sigmoid
        output_pose_detach = pred_corners_undetach = 0

        dec_out_poses = []
        dec_out_logits = []
        dec_out_refs = []
        dec_out_pred_corners = []

        for layer_id, layer in enumerate(self.layers):
            refpoint_pose_input = refpoint_pose[:, :, None] 
            refpoint_only_pose = refpoint_pose[:, :, 1:]
            pose_query_sine_embed = self.sine_embedding(refpoint_only_pose)
            pose_query_pos = self.half_pose_ref_point_head(pose_query_sine_embed)
            
            output = layer(
                tgt_pose = output,
                tgt_pose_query_pos = pose_query_pos,
                tgt_pose_reference_points = refpoint_pose_input,
                attn_mask=attn_mask,
                
                memory = memory,
                memory_spatial_shapes = spatial_shapes,
            )
            
            

            output_pose = output[:, :, 1:]
            output_instance = output[:, :, 0]

            # iteration
            if layer_id == 0:
                # Initial bounding box predictions with inverse sigmoid refinement
                pre_poses = F.sigmoid(pre_pose_head(output_pose) + inverse_sigmoid(refpoint_only_pose))
                pre_scores = class_head[0](output_instance)
                ref_pose_initial = pre_poses.detach()

            # Refine bounding box corners using FDR, integrating previous layer's corrections
            pred_corners = pose_head[layer_id](output_pose + output_pose_detach) + pred_corners_undetach
            refpoint_pose_without_center = distance2pose(
                ref_pose_initial, integral(pred_corners, project), reg_scale
            )

            # center of pose
            refpoint_center_pose = torch.mean(refpoint_pose_without_center, dim=2, keepdim=True)
            refpoint_pose = torch.cat([refpoint_center_pose, refpoint_pose_without_center], dim=2)
            
            
            
                # refpoint_pose = z.detach() if not self.training else z


            if self.training or layer_id ==self.eval_idx:
                score = class_head[layer_id](output_instance)
                logit = lqe_head[layer_id](score, refpoint_pose_without_center, feat_lqe)
                
                # -------------------- energy-based refinement loop (minimal) --------------------
                if self.use_energy_refinement and layer_id == self.num_layers - 1:
                    n_pred_corners = pred_corners.size()[-1]
                    n_refpoint_pose = refpoint_pose_without_center.size()[-1]
                    n_logit        = logit.size()[-1]
                    # build z
                    z = torch.cat(
                        (
                            torch.cat((pred_corners, refpoint_pose_without_center), dim=-1),
                            logit[..., None, :].repeat((1, 1, 1, (n_pred_corners + n_refpoint_pose) // n_logit))
                        ),
                        dim=-2
                    )

                    # ensure z participates in autograd
                    if self.training:
                        # keep graph if you want higher-order grads through the refinement
                        z.requires_grad_(True)
                    else:
                        # during eval we usually don't need grads to flow back into the base network
                        z = z.detach().requires_grad_(True)

                    condition = tuple(m.detach() for m in memory)

                    # breakpoint()
                    if isinstance(self.energy_steps, DictConfig):
                        self.energy_steps = OmegaConf.to_container(self.energy_steps, resolve=True)

                    energy_reg_loss = torch.zeros(1, device=z.device, dtype=z.dtype)  # accumulates reg term
                    E_prev = None
                    lambda_energy = getattr(self, "energy_decrease_weight", 1e-2)  # tune this

                    for i in range(self._resolve_energy_steps(is_training=self.training)):
                        E_raw = self.energy_head(
                            tgt_pose=z,
                            tgt_pose_query_pos=pose_query_pos,
                            tgt_pose_reference_points=refpoint_pose_input,
                            attn_mask=attn_mask,
                            memory=condition,
                            memory_spatial_shapes=spatial_shapes
                        )

                        # 1️⃣ Compute a safe energy term (same as your original)
                        E_neg = -E_raw
                        E_safe = torch.logsumexp(E_neg.view(E_neg.shape[0], -1), dim=1)  # shape: (batch,)
                        E_safe = torch.clamp(E_safe, -50, 50)

                        # ---------- NEW: compute per-iteration decrease regulariser ----------
                        # reg_i = ReLU( E_t - stop_gradient(E_{t-1}) )  (per-example)
                        if E_prev is not None:
                            # stop gradient on previous energy so only E_safe contributes gradients
                            per_example_reg = torch.relu(E_safe - E_prev.detach())  # shape (batch,)
                            # aggregate (mean) and weight it
                            energy_reg_loss = energy_reg_loss + lambda_energy * per_example_reg.mean()
                        # set previous energy for next iteration
                        E_prev = E_safe
                        # --------------------------------------------------------------------

                        # 3️⃣ Compute gradient for z
                        grad_z = torch.autograd.grad(E_safe.sum(), z, create_graph=self.training)[0]

                        if torch.isnan(grad_z).any():
                            print("Warning: NaN in grad_z detected!")

                        self.energy_step_size.data = self.energy_step_size.data.to(dtype=z.dtype, device=z.device)
                        if self.training:
                            noise = torch.randn_like(z) * self.noise_scale
                        else:
                            noise = 0.0
                        z = z - self.energy_step_size * grad_z + noise

                        if self.loss_all_steps and i < self._resolve_energy_steps(is_training=self.training) - 1:
                            # Re-extract components after each refinement step for loss computation
                            pred_corners = z[..., :-1, :n_pred_corners]
                            refpoint_pose_without_center = z[..., :-1, n_pred_corners:]
                            logit = z[..., -1, 0:2]
                            dec_out_logits.append(logit)
                            dec_out_poses.append(refpoint_pose_without_center)
                            dec_out_pred_corners.append(pred_corners)
                            dec_out_refs.append(ref_pose_initial)
                        
                        # breakpoint()
                    pred_corners = z[..., :-1, :n_pred_corners]
                    refpoint_pose_without_center = z[..., :-1, n_pred_corners:]
                    logit = z[..., -1, 0:2]
                    # breakpoint()
                    self.layer_loss = energy_reg_loss

                dec_out_logits.append(logit)
                dec_out_poses.append(refpoint_pose_without_center)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_pose_initial)
                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            if self.training:
                refpoint_pose = refpoint_pose.detach()
                output_pose_detach = output_pose.detach()
            else:
                refpoint_pose = refpoint_pose
                output_pose_detach = output_pose

        return (
            torch.stack(dec_out_poses), 
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_poses,
            pre_scores,            
        )



class Transformer(nn.Module):
    # Fine-Distribution-Refinement Transformer from D-FINE
    def __init__(
        self, 
        hidden_dim=256, 
        nhead=8, 
        num_queries=300, 
        num_decoder_layers=6, 
        dim_feedforward=2048, 
        dropout=0.0,
        activation="relu", 
        normalize_before=False,
        return_intermediate_dec=False, 
        num_feature_levels=1,
        enc_n_points=4,
        dec_n_points=4,
        learnable_tgt_init=False,
        two_stage_type='no',
        num_classes=2, 
        aux_loss=True,
        dec_pred_class_embed_share=False,
        dec_pred_pose_embed_share=False,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        cls_no_bias = False,
        num_body_points=17,
        # new parameters
        feat_strides=None,
        eval_spatial_size=None,
        reg_max=32,
        reg_scale=4.0,
        use_kan=False, 
        kan_grid=3,
        use_modulation=False,
        use_region_sampling=False,
        region_kernel_size=1,
        use_global_context=False,
        use_grouped_offsets=False,
        num_groups=1,
        use_grid_attention=False,
        grid_num_points=16,
        use_grid_offsets=False,
        use_grid_fusion=True,
        use_energy_refinement = False,
        energy_steps = 3,
        energy_step_size = 1.0,
        energy_hidden = 256,
        energy_n_layers = 2,
        freeze_network=False,
        noise_scale = 0.01,
        energy_in_dim=68,
        energy_out_dim=1,
        loss_all_steps=False,
        energy_decrease_weight=0.0,
        ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_decoder_layers = num_decoder_layers
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        self.energy_decrease_weight = energy_decrease_weight
        
   
        decoder_layer = DeformableTransformerDecoderLayer(hidden_dim, dim_feedforward, dropout,
                                                          activation, num_feature_levels, nhead,
                                                          dec_n_points, use_kan=use_kan, kan_grid=kan_grid, use_modulation=use_modulation, 
                                                          use_region_sampling=use_region_sampling, region_kernel_size=region_kernel_size,
                                                          use_global_context=use_global_context, use_grouped_offsets=use_grouped_offsets, num_groups=num_groups,
                                                          use_grid_attention=use_grid_attention, grid_num_points=grid_num_points, use_grid_offsets=use_grid_offsets, use_grid_fusion=use_grid_fusion)
        if use_energy_refinement:
            print(">>> Initializing energy-based refinement layer")
            self.energy_layer = DeformableTransformerDecoderLayer(hidden_dim, dim_feedforward, dropout,
                                                          activation, num_feature_levels, nhead,
                                                          dec_n_points, use_kan=use_kan, kan_grid=kan_grid, use_modulation=use_modulation, 
                                                          use_region_sampling=use_region_sampling, region_kernel_size=region_kernel_size,
                                                          use_global_context=use_global_context, use_grouped_offsets=use_grouped_offsets, num_groups=num_groups,
                                                          use_grid_attention=use_grid_attention, grid_num_points=grid_num_points, use_grid_offsets=use_grid_offsets, use_grid_fusion=use_grid_fusion,
                                                          is_energy=True, energy_in_dim=energy_in_dim, energy_out_dim=energy_out_dim)
        else:
            self.energy_layer = None

        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers,
                                        return_intermediate=return_intermediate_dec,
                                        hidden_dim=hidden_dim,
                                        num_body_points=num_body_points, use_energy_refinement=use_energy_refinement,
                                        energy_steps=energy_steps, energy_step_size=energy_step_size,
                                        energy_hidden=energy_hidden, energy_n_layers=energy_n_layers, energy_layer=self.energy_layer, noise_scale=noise_scale, loss_all_steps=loss_all_steps, energy_decrease_weight=self.energy_decrease_weight)
        self.layer_loss = torch.zeros(1, dtype=torch.float32)
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries
        
        # shared prior between instances
        self.num_body_points = num_body_points
        self.keypoint_embedding = nn.Embedding(num_body_points, self.hidden_dim)
        self.instance_embedding = nn.Embedding(1, self.hidden_dim)
        
        self.learnable_tgt_init = learnable_tgt_init
        if learnable_tgt_init:
            self.tgt_embed = nn.Embedding(self.num_queries, self.hidden_dim)
            # self.register_buffer("tgt_embed", torch.zeros(self.num_queries, hidden_dim))
        else:
            self.tgt_embed = None

        self.label_enc = nn.Embedding(80 + 1, hidden_dim)
        self.pose_enc = nn.Embedding(num_body_points, hidden_dim)
            
        self._reset_parameters()

        # for two stage
        self.two_stage_type = two_stage_type
        if two_stage_type in ['standard']:
            # anchor selection at the output of encoder
            self.enc_output = nn.Linear(hidden_dim, hidden_dim)
            self.enc_output_norm = nn.LayerNorm(hidden_dim)
        self.enc_out_class_embed = None
        self.enc_pose_embed = None

        # prepare class
        _class_embed = nn.Linear(hidden_dim, num_classes, bias=(not cls_no_bias))
        if not cls_no_bias:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            _class_embed.bias.data = torch.ones(self.num_classes) * bias_value

        _pre_point_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        nn.init.constant_(_pre_point_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_pre_point_embed.layers[-1].bias.data, 0)

        _point_embed = MLP(hidden_dim, hidden_dim, 2 * (reg_max + 1), 3)
        nn.init.constant_(_point_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_point_embed.layers[-1].bias.data, 0)

        _lqe_embed = LQE(4, 256, 2, num_body_points)
        
        if dec_pred_class_embed_share:
            class_embed_layerlist = [_class_embed for i in range(num_decoder_layers)]
            lqe_embed_layerlist = [_lqe_embed for i in range(num_decoder_layers)]            
        else:
            class_embed_layerlist = [copy.deepcopy(_class_embed) for i in range(num_decoder_layers)]
            lqe_embed_layerlist = [copy.deepcopy(_lqe_embed) for i in range(num_decoder_layers)]
        
        if dec_pred_pose_embed_share:
            pose_embed_layerlist = [_point_embed for i in range(num_decoder_layers)]
        else:
            pose_embed_layerlist = [copy.deepcopy(_point_embed) for i in range(num_decoder_layers)]

        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.pose_embed = nn.ModuleList(pose_embed_layerlist)
        self.lqe_embed = nn.ModuleList(lqe_embed_layerlist)
        self.pre_pose_embed = _pre_point_embed
        self.integral = Integral(reg_max)

        self.up = nn.Parameter(torch.tensor([1/2]), requires_grad=False)
        self.reg_max = reg_max
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        self.deploy = False


        # two stage
        _keypoint_embed = MLP(hidden_dim, 2*hidden_dim, 2*num_body_points, 4)
        nn.init.constant_(_keypoint_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_keypoint_embed.layers[-1].bias.data, 0)
        
        if two_stage_bbox_embed_share:
            self.enc_pose_embed = _keypoint_embed
        else:
            self.enc_pose_embed = copy.deepcopy(_keypoint_embed)

        if two_stage_class_embed_share:
            self.enc_out_class_embed = _class_embed
        else:
            self.enc_out_class_embed = copy.deepcopy(_class_embed)

        # for inference
        self.feat_strides = feat_strides
        self.eval_spatial_size = eval_spatial_size
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer("anchors", anchors)
            self.register_buffer("valid_mask", valid_mask)

        if freeze_network:
            print(">>> Freezing Deformable Transformer Network Parameters")
            self._freeze_parameters(self)

            # # Unfreeze only energy_layer (if it exists)
            # if hasattr(self.decoder, "energy_layer") and self.decoder.energy_layer is not None:
            #     print(">>> Unfreezing energy_layer parameters")
            #     for p in self.decoder.energy_layer.parameters():
            #         p.requires_grad = True

    def _freeze_parameters(self, module):
        for p in module.parameters():
            p.requires_grad = False

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        # get projection features
        proj_feats = feats

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        split_sizes = []

        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            split_sizes.append(h*w)

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes, split_sizes

    def _generate_anchors(self, spatial_shapes=None, device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=device),
                                            indexing='ij')
            grid = torch.stack([grid_x, grid_y], -1) # H_, W_, 2

            grid = (grid.unsqueeze(0).expand(1, -1, -1, -1) + 0.5) /  torch.tensor([W_, H_], dtype=torch.float32, device=device)
                
            lvl_anchor = grid.view(1, -1, 2)
            anchors.append(lvl_anchor)
        anchors = torch.cat(anchors, 1)
        valid_mask = ((anchors > 0.01) & (anchors < 0.99)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors)) # unsigmoid
        return anchors, ~valid_mask
        
    def convert_to_deploy(self):
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        self.lqe_embed = nn.ModuleList(
            [nn.Identity()] * (self.dec_layers-1) + [self.lqe_embed[self.dec_layers-1]]
        )
        self.deploy = True

    def forward(self, feats, targets, samples=None):
        """
        Input:
            - feats: List of multi features [bs, ci, hi, wi]
            
        """
        # input projection and embedding
        memory, spatial_shapes, split_sizes = self._get_encoder_input(feats)

        # Two-stage starts here
        if self.training:
            output_proposals, valid_mask = self._generate_anchors(spatial_shapes, memory.device)
            output_memory = memory.masked_fill(valid_mask, float(0))
            output_proposals = output_proposals.repeat(memory.size(0), 1, 1)
        else: 
            output_proposals = self.anchors.repeat(memory.size(0), 1, 1)
            output_memory = memory.masked_fill(self.valid_mask, float(0))

        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        # top-k select index
        topk = self.num_queries
        enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
        topk_idx = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]
        
        # calculate K, e.g., 17 for COCO, points for keypoint
        topk_memory = output_memory.gather(
            dim=1, index=topk_idx.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])
        )
        topk_anchors = output_proposals.gather(
            dim=1, index=topk_idx.unsqueeze(-1).repeat(1, 1, 2)
            )

        bs, nq = topk_memory.shape[:2]
        delta_unsig_keypoint = self.enc_pose_embed(topk_memory).reshape(bs, nq, self.num_body_points, 2)
        enc_outputs_pose_coord = F.sigmoid(delta_unsig_keypoint + topk_anchors.unsqueeze(-2))
        enc_outputs_center_coord = torch.mean(enc_outputs_pose_coord, dim=2, keepdim=True)
        enc_outputs_pose_coord = torch.cat([enc_outputs_center_coord, enc_outputs_pose_coord], dim=2)
        refpoint_pose_sigmoid = enc_outputs_pose_coord.detach()

        interm_class = torch.gather(enc_outputs_class_unselected, 1, 
            topk_idx.unsqueeze(-1).repeat(1, 1, enc_outputs_class_unselected.shape[-1])
            ) if self.training else None 

        # combine pose embedding
        if self.learnable_tgt_init:
            tgt = self.tgt_embed.weight.unsqueeze(0).repeat([memory.shape[0], 1, 1]).unsqueeze(-2)
            # tgt = self.tgt_embed.unsqueeze(0).tile([memory.shape[0], 1, 1]).unsqueeze(-2)
        else:
            tgt = topk_memory.detach().unsqueeze(-2)
        # query construction
        tgt_pose = self.keypoint_embedding.weight[None, None].repeat(1, topk, 1, 1).expand(bs, -1, -1, -1) + tgt
        tgt_global = self.instance_embedding.weight[None, None].repeat(1, topk, 1, 1).expand(bs, -1, -1, -1)
        tgt_pose = torch.cat([tgt_global, tgt_pose], dim=2)
        # Two-stage ends here

        # Denoising starts here
        if self.training and targets is not None:
            input_query_label, input_query_pose, attn_mask, dn_meta = prepare_for_cdn(
                dn_args=(targets, 20, 0.5),
                training=self.training,
                num_queries=self.num_queries,
                hidden_dim=self.hidden_dim,
                num_classes=80,
                label_enc=self.label_enc,
                pose_enc=self.pose_enc,
                num_keypoints=self.num_body_points,
                img_dim=samples.shape[-2:],
                device=feats[0].device
                )
            tgt_pose = torch.cat([input_query_label, tgt_pose], dim=1)
            refpoint_pose_sigmoid = torch.cat([input_query_pose.sigmoid(), refpoint_pose_sigmoid], dim=1)
        else:
            input_query_label = input_query_pose = attn_mask = dn_meta = None
        # Denoising ends here

        # preprocess memory for MSDeformableLineAttention
        value = memory.unflatten(2, (self.nhead, -1)) # (bs, \sum{hxw}, n_heads, d_model//n_heads)
        value = value.permute(0, 2, 3, 1).flatten(0, 1).split(split_sizes, dim=-1)
        
        if not hasattr(self, "project"):
            project = weighting_function(self.reg_max, self.up, self.reg_scale)
        else:
            project = self.project

        (
            out_poses, 
            out_logits, 
            out_corners, 
            out_references, 
            out_pre_poses,
            out_pre_scores,) = self.decoder(
                tgt=tgt_pose,
                memory=value,  
                refpoints_sigmoid=refpoint_pose_sigmoid, 
                spatial_shapes=spatial_shapes,
                attn_mask=attn_mask,
                pre_pose_head=self.pre_pose_embed,
                pose_head=self.pose_embed,
                class_head=self.class_embed,
                lqe_head=self.lqe_embed,
                feat_lqe=feats[0],
                # new arguments
                up=self.up,
                reg_max=self.reg_max,
                reg_scale=self.reg_scale,
                integral=self.integral,
                project=project,
                )
        self.layer_loss = self.decoder.layer_loss
        if not self.deploy:
            out_poses = out_poses.flatten(-2)

        if self.training and dn_meta is not None:
            # flattenting (L, bs, nq, np, 2) -> (L, bs, nq, np * 2)
            out_pre_poses = out_pre_poses.flatten(-2)
            
            dn_out_poses, out_poses = torch.split(out_poses,[dn_meta['pad_size'], self.num_queries], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, [dn_meta['pad_size'], self.num_queries], dim=2)

            dn_out_corners, out_corners = torch.split(out_corners, [dn_meta['pad_size'], self.num_queries], dim=2)
            dn_out_refs, out_refs = torch.split(out_references, [dn_meta['pad_size'], self.num_queries], dim=2)

            dn_out_pre_poses, out_pre_poses = torch.split(out_pre_poses,[dn_meta['pad_size'], self.num_queries], dim=1)
            dn_out_pre_scores, out_pre_scores = torch.split(out_pre_scores, [dn_meta['pad_size'], self.num_queries], dim=1)

        out = {'pred_logits': out_logits[-1], 'pred_keypoints': out_poses[-1]}

        if self.training and self.aux_loss:
            out.update({
                "pred_corners": out_corners[-1],
                "ref_points": out_refs[-1],
                "up": self.up,
                "reg_scale": self.reg_scale,
                'reg_max': self.reg_max
                })

            out['aux_outputs'] = self._set_aux_loss2(
                out_logits[:-1], 
                out_poses[:-1],
                out_corners[:-1],
                out_refs[:-1],
                out_corners[-1],
                out_logits[-1],
                )
            # prepare intermediate outputs
            out['aux_interm_outputs'] = [{'pred_logits': interm_class, 'pred_keypoints': enc_outputs_pose_coord[:, :, 1:].flatten(-2)}]
            out['aux_pre_outputs'] =  {'pred_logits': out_pre_scores, 'pred_keypoints': out_pre_poses}
            
            if dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss2(
                    dn_out_logits, 
                    dn_out_poses, 
                    dn_out_corners, 
                    dn_out_refs, 
                    dn_out_corners[-1], 
                    dn_out_logits[-1]
                    )
                out['dn_aux_pre_outputs'] =  {'pred_logits': dn_out_pre_scores, 'pred_keypoints': dn_out_pre_poses}
                out['dn_meta'] = dn_meta

        return out #hs_pose, refpoint_pose, mix_refpoint, mix_embedding

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_keypoints):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_keypoints': c}
                for a, c in zip(outputs_class, outputs_keypoints)]

    @torch.jit.unused
    def _set_aux_loss2(
        self,
        outputs_class,
        outputs_keypoints,
        outputs_corners,
        outputs_ref,
        teacher_corners=None,
        teacher_logits=None,
    ):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {
                "pred_logits": a,
                "pred_keypoints": b,
                "pred_corners": c,
                "ref_points": d,
                "teacher_corners": teacher_corners,
                "teacher_logits": teacher_logits,
            }
            for a, b, c, d in zip(outputs_class, outputs_keypoints, outputs_corners, outputs_ref)
        ]