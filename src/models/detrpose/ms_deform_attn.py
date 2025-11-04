# ------------------------------------------------------------------------------------------------
# Deformable DETR (extended with optional deformable grid attention + fusion)
# - Backwards-compatible: defaults preserve original behavior
# ------------------------------------------------------------------------------------------------

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

import torch
import torch.nn.functional as F

def soft_grid_sample(value, grid, align_corners=False):
    """
    High-fidelity differentiable reimplementation of F.grid_sample for 2D bilinear interpolation.

    value: (B, C, H, W)
    grid: (B, Len_q, P, 2), normalized coordinates in [-1, 1]
    returns: (B, C, Len_q, P)
    """
    B, C, H, W = value.shape
    _, Len_q, P, _ = grid.shape

    # Convert from [-1, 1] to [0, H-1] and [0, W-1]
    if align_corners:
        x = ((grid[..., 0] + 1) / 2) * (W - 1)
        y = ((grid[..., 1] + 1) / 2) * (H - 1)
    else:
        x = ((grid[..., 0] + 1) * W - 1) / 2
        y = ((grid[..., 1] + 1) * H - 1) / 2

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    # Clip to valid range
    x0_clipped = x0.clamp(0, W - 1)
    x1_clipped = x1.clamp(0, W - 1)
    y0_clipped = y0.clamp(0, H - 1)
    y1_clipped = y1.clamp(0, H - 1)

    # Compute interpolation weights
    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0.float())
    wc = (x - x0.float()) * (y1.float() - y)
    wd = (x - x0.float()) * (y - y0.float())

    # Flatten value for fast gather
    value_flat = value.reshape(B, C, H * W)

    def gather(b, x_idx, y_idx):
        idx = (y_idx * W + x_idx).reshape(B, -1)
        gathered = torch.gather(value_flat, 2, idx.unsqueeze(1).expand(-1, C, -1))
        return gathered.reshape(B, C, Len_q, P)

    Ia = gather(B, x0_clipped, y0_clipped)
    Ib = gather(B, x0_clipped, y1_clipped)
    Ic = gather(B, x1_clipped, y0_clipped)
    Id = gather(B, x1_clipped, y1_clipped)

    # Weighted sum (broadcasted)
    out = Ia * wa.unsqueeze(1) + Ib * wb.unsqueeze(1) + Ic * wc.unsqueeze(1) + Id * wd.unsqueeze(1)

    # Zero padding where samples fall outside [-1, 1]
    mask = ((x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)).float()
    out = out * mask.unsqueeze(1)

    return out


def bilinear_sample_pytorch(value, sampling_grid, align_corners=False):
    """
    Vectorized bilinear sampler written in plain PyTorch.
    value: (B, C, H, W)
    sampling_grid: (B, Len_q, P, 2) with coords in [-1, 1] (x, y)
    returns: (B, C, Len_q, P)
    """
    B, C, H, W = value.shape
    _, Len_q, P, _ = sampling_grid.shape

    # convert normalized [-1,1] coords to pixel coords
    if align_corners:
        px = (sampling_grid[..., 0] + 1.0) * 0.5 * (W - 1)  # (B, Len_q, P)
        py = (sampling_grid[..., 1] + 1.0) * 0.5 * (H - 1)
    else:
        # grid_sample's default align_corners=False mapping:
        px = (sampling_grid[..., 0] + 1.0) * 0.5 * (W - 1)
        py = (sampling_grid[..., 1] + 1.0) * 0.5 * (H - 1)

    # floor and ceil coords
    x0 = torch.floor(px).clamp(0, W - 1)
    x1 = (x0 + 1).clamp(0, W - 1)
    y0 = torch.floor(py).clamp(0, H - 1)
    y1 = (y0 + 1).clamp(0, H - 1)

    # fractional part
    wa = (x1 - px) * (y1 - py)  # top-left weight
    wb = (x1 - px) * (py - y0)  # bottom-left
    wc = (px - x0) * (y1 - py)  # top-right
    wd = (px - x0) * (py - y0)  # bottom-right

    # convert to long indices for gather
    x0_l = x0.long()
    x1_l = x1.long()
    y0_l = y0.long()
    y1_l = y1.long()

    # flatten spatial dims
    value_flat = value.view(B, C, H * W)  # (B, C, HW)
    # compute linear indices
    idx_a = (y0_l * W + x0_l).reshape(B, -1)
    idx_b = (y1_l * W + x0_l).reshape(B, -1)
    idx_c = (y0_l * W + x1_l).reshape(B, -1)
    idx_d = (y1_l * W + x1_l).reshape(B, -1)


    # expand indices to channels for gather
    # gather needs index shape (B, C, L) when gathering along last dim
    def gather_by_index(value_flat, idx):
        # value_flat: (B, C, HW)
        B, C, HW = value_flat.shape
        L = idx.shape[1]  # Len_q * P
        idx_expand = idx.unsqueeze(1).expand(-1, C, -1)  # (B, C, L)
        gathered = torch.gather(value_flat, 2, idx_expand)  # (B, C, L)
        return gathered.view(B, C, Len_q, P)  # reshape to (B, C, Len_q, P)

    Ia = gather_by_index(value_flat, idx_a)
    Ib = gather_by_index(value_flat, idx_b)
    Ic = gather_by_index(value_flat, idx_c)
    Id = gather_by_index(value_flat, idx_d)

    # reshape weights to (B, 1, Len_q, P)
    wa = wa.view(B, 1, Len_q, P)
    wb = wb.view(B, 1, Len_q, P)
    wc = wc.view(B, 1, Len_q, P)
    wd = wd.view(B, 1, Len_q, P)

    out = wa * Ia + wb * Ib + wc * Ic + wd * Id  # (B, C, Len_q, P)
    return out


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights,
                                sampling_modulation=None, region_kernel_size=1, is_energy=False):
    """
    Core multi-scale deformable attention routine using grid_sample.
    - value: list of per-level tensors; each element shaped (N*Mprime, C_per_sample, H*W)
             (we unflatten last dim to (H,W) for grid_sample)
    - sampling_locations: (N, Len_q, Mprime, L, P, 2) normalized coords [0,1]
    - attention_weights: (N, Len_q, Mprime, L, P)
    - sampling_modulation: optional (N, Len_q, Mprime, L, P)
    - region_kernel_size: if >1, apply avg_pool2d to the per-level unflattened maps before sampling
    Returns:
        output: (N, Len_q, Mprime * C_per_sample) -> typically equals (N, Len_q, d_model)
    """
    _, D_, _ = value[0].shape
    N_, Len_q, Mprime, L_, P_, _ = sampling_locations.shape

    sampling_grids = 2 * sampling_locations - 1  # to [-1,1] for grid_sample
    sampling_grids = sampling_grids.transpose(1, 2).flatten(0, 1)  # (N*Mprime, Len_q, L, P, 2)

    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value[lid_].unflatten(2, (H_, W_))  # (N*Mprime, C, H, W)

        if region_kernel_size is not None and region_kernel_size > 1:
            pad = region_kernel_size // 2
            value_l_ = F.avg_pool2d(value_l_, kernel_size=region_kernel_size, stride=1, padding=pad)

        sampling_grid_l_ = sampling_grids[:, :, lid_]  # (N*Mprime, Len_q, P, 2)
        # sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros', align_corners=False)
        # sampling_value_l_ = bilinear_sample_pytorch(value_l_, sampling_grid_l_, align_corners=False)
        if is_energy:
            sampling_value_l_ = soft_grid_sample(value_l_, sampling_grid_l_, align_corners=False)
        else:
            sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros', align_corners=False)


        # sampling_value_l_: (N*Mprime, C, Len_q, P)
        sampling_value_list.append(sampling_value_l_)

    sampled = torch.cat(sampling_value_list, dim=-1)  # (N*Mprime, C, Len_q, L_*P_)

    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * Mprime, 1, Len_q, L_ * P_)

    if sampling_modulation is not None:
        modulation = sampling_modulation.transpose(1, 2).reshape(N_ * Mprime, 1, Len_q, L_ * P_)
        sampled = sampled * modulation  # broadcast over channel dim

    output = (sampled * attention_weights).sum(-1).view(N_, Mprime * D_, Len_q)
    return output.transpose(1, 2)  # (N, Len_q, Mprime * D_)


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, use_4D_normalizer=False,
                 # optional flags (defaults keep original behavior)
                 use_modulation=False, use_region_sampling=False, region_kernel_size=1,
                 use_global_context=False, use_grouped_offsets=False, num_groups=1,
                 # new grid-attention options:
                 use_grid_attention=False, grid_num_points=16, use_grid_offsets=False,
                 use_grid_fusion=True, is_energy=False):
        """
        Multi-Scale Deformable Attention with optional branches:
        - local deformable attention (original)
        - optional deformable grid attention (coarse global grid per level)
        - optional fusion (linear projection) to combine branches

        Defaults: all new options are disabled, so module behaves exactly like original MSDeformAttn.
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads

        # core params
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.is_energy = is_energy

        # options previously added
        self.use_modulation = bool(use_modulation)
        self.use_region_sampling = bool(use_region_sampling)
        self.region_kernel_size = int(region_kernel_size)
        self.use_global_context = bool(use_global_context)
        self.use_grouped_offsets = bool(use_grouped_offsets)
        self.num_groups = int(num_groups) if self.use_grouped_offsets else 1

        # Grid attention options
        self.use_grid_attention = bool(use_grid_attention)
        self.grid_num_points = int(grid_num_points) if self.use_grid_attention else 0
        if self.use_grid_attention:
            # require perfect square for simple grid layout
            gs = int(math.sqrt(self.grid_num_points))
            if gs * gs != self.grid_num_points:
                raise ValueError("grid_num_points must be a perfect square (e.g., 4,9,16). got {}".format(self.grid_num_points))
            self._grid_size = gs  # gs x gs grid per level
        self.use_grid_offsets = bool(use_grid_offsets) if self.use_grid_attention else False
        self.use_grid_fusion = bool(use_grid_fusion) if self.use_grid_attention else False

        # local deformable branch predictors (offsets + attention)
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2 * self.num_groups)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)

        # modulation branch (local deformable) - optional (bias-free linear + LayerNorm + learnable bias)
        if self.use_modulation:
            self.sampling_modulation = nn.Linear(d_model, n_heads * n_levels * n_points * self.num_groups, bias=False)
            self.mod_layernorm = nn.LayerNorm([self.n_levels, self.n_points])
            mshape = (1, 1, self.n_heads * self.num_groups, self.n_levels, self.n_points)
            self.mod_bias = nn.Parameter(torch.ones(*mshape, dtype=torch.float32) * 2.0)
        else:
            self.sampling_modulation = None
            self.mod_layernorm = None
            self.mod_bias = None

        # optional global context and elementwise gate
        if self.use_global_context:
            self.global_proj = nn.Linear(_d_per_head * n_levels, d_model)
            self.global_gate = nn.Linear(d_model, d_model)  # elementwise gate
        else:
            self.global_proj = None
            self.global_gate = None

        # grid branch predictors (if enabled)
        if self.use_grid_attention:
            # attention for grid branch
            self.grid_attention_weights = nn.Linear(d_model, n_heads * n_levels * self.grid_num_points)
            if self.use_grid_offsets:
                self.grid_offsets = nn.Linear(d_model, n_heads * n_levels * self.grid_num_points * 2 * self.num_groups)
            else:
                self.grid_offsets = None

            # fuse projection when both branches are used
            if self.use_grid_fusion:
                self.grid_fuse_proj = nn.Linear(2 * d_model, d_model)
            else:
                self.grid_fuse_proj = None

            # prepare base grid coordinates per-level (created on first forward and cached)
            self.register_buffer('_base_grid_placeholder', torch.tensor([0.0]), persistent=False)
        else:
            self.grid_attention_weights = None
            self.grid_offsets = None
            self.grid_fuse_proj = None

        self.use_4D_normalizer = use_4D_normalizer

        self._reset_parameters()

    def _reset_parameters(self):
        # sampling offsets init (same pattern as original)
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]) \
                        .view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)

        if self.num_groups > 1:
            grid_init = grid_init.unsqueeze(3).repeat(1, 1, 1, self.num_groups, 1)
            grid_init = grid_init.view(self.n_heads, self.n_levels, self.n_points * self.num_groups, 2)

        with torch.no_grad():
            try:
                self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
            except Exception:
                constant_(self.sampling_offsets.bias, 0.)

        if self.n_points % 4 != 0 and self.num_groups == 1:
            constant_(self.sampling_offsets.bias, 0.)

        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)

        # modulation init
        if self.sampling_modulation is not None:
            with torch.no_grad():
                self.sampling_modulation.weight.zero_()
            # mod_bias already set to 2.0 on creation

        # global context init
        if self.global_proj is not None:
            xavier_uniform_(self.global_proj.weight.data)
            constant_(self.global_proj.bias.data, 0.)
        if self.global_gate is not None:
            with torch.no_grad():
                self.global_gate.weight.zero_()
                self.global_gate.bias.fill_(2.0)  # gate starts open ~0.88

        # grid branch init
        if self.grid_attention_weights is not None:
            constant_(self.grid_attention_weights.weight.data, 0.)
            constant_(self.grid_attention_weights.bias.data, 0.)
        if self.grid_offsets is not None:
            with torch.no_grad():
                self.grid_offsets.weight.zero_()
                # start with zero offsets so base grid is used initially
                constant_(self.grid_offsets.bias.data, 0.)
        if self.grid_fuse_proj is not None:
            xavier_uniform_(self.grid_fuse_proj.weight.data)
            constant_(self.grid_fuse_proj.bias.data, 0.)

    # helper: build base grid coords per-level (cached)
    def _build_base_grid(self, input_spatial_shapes, device):
        # returns tensor of shape (n_levels, P_grid, 2) with normalized coords in [0,1]
        grids = []
        for lid_, (H_, W_) in enumerate(input_spatial_shapes):
            gs = self._grid_size
            # center coordinates of a gs x gs grid in normalized [0,1]
            xs = torch.linspace(0.5 / gs, 1.0 - 0.5 / gs, steps=gs, device=device)
            ys = torch.linspace(0.5 / gs, 1.0 - 0.5 / gs, steps=gs, device=device)
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')
            coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # (P_grid, 2): x,y
            grids.append(coords)
        return torch.stack(grids, dim=0)  # (n_levels, P_grid, 2)

    def forward(self, query, reference_points, value, input_spatial_shapes):
        """
        query: (N, Len_q, C)
        reference_points: (N, Len_q, n_levels, 2) or (N, Len_q, n_levels, 4)
        value: list of per-level tensors where each value[l] is shaped (N * n_heads, d_per_head, H_l * W_l)
        input_spatial_shapes: list/tuple of (H_l, W_l) pairs
        """
        N, Len_q, _ = query.shape
        d_per_head = self.d_model // self.n_heads

        # optional global context conditioning
        if self.use_global_context:
            pooled_per_level = []
            for lid_, (H_, W_) in enumerate(input_spatial_shapes):
                v = value[lid_]
                v = v.view(N, self.n_heads, d_per_head, H_, W_)
                pooled = v.mean(dim=(1, 3, 4))  # (N, d_per_head)
                pooled_per_level.append(pooled)
            pooled_cat = torch.cat(pooled_per_level, dim=-1)  # (N, d_per_head * n_levels)
            global_ctx = self.global_proj(pooled_cat)  # (N, d_model)
            gate = torch.sigmoid(self.global_gate(query))  # (N, Len_q, d_model)
            conditioning = query + gate * global_ctx.unsqueeze(1)
        else:
            conditioning = query

        # ---------- Local deformable branch (original) ----------
        sampling_offsets = self.sampling_offsets(conditioning)
        if self.num_groups > 1:
            sampling_offsets = sampling_offsets.view(N, Len_q, self.n_heads, self.n_levels, self.n_points, self.num_groups, 2)
            sampling_offsets = sampling_offsets.permute(0, 1, 2, 5, 3, 4, 6).reshape(N, Len_q, self.n_heads * self.num_groups, self.n_levels, self.n_points, 2)
        else:
            sampling_offsets = sampling_offsets.view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)

        attention_weights = self.attention_weights(conditioning).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        if self.num_groups > 1:
            attention_weights = attention_weights.unsqueeze(3).repeat(1, 1, 1, self.num_groups, 1, 1)
            attention_weights = attention_weights.view(N, Len_q, self.n_heads * self.num_groups, self.n_levels, self.n_points)

        # modulation (local branch)
        sampling_modulation = None
        if self.sampling_modulation is not None:
            mod = self.sampling_modulation(conditioning)
            if self.num_groups > 1:
                mod = mod.view(N, Len_q, self.n_heads, self.n_levels, self.n_points, self.num_groups)
                mod = mod.permute(0, 1, 2, 5, 3, 4).reshape(N, Len_q, self.n_heads * self.num_groups, self.n_levels, self.n_points)
            else:
                mod = mod.view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
            mod = self.mod_layernorm(mod)
            mod = mod + self.mod_bias
            sampling_modulation = torch.sigmoid(mod)

        # compute sampling locations for local branch
        reference_points_t = torch.transpose(reference_points, 2, 3).flatten(1, 2)  # (N, Len_q, n_levels, 2)
        if reference_points_t.shape[-1] == 2:
            offset_normalizer = torch.tensor(input_spatial_shapes, device=query.device)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.n_levels, 1, 2)
            sampling_locations = reference_points_t[:, :, None, :, None, :] + sampling_offsets / offset_normalizer
        elif reference_points_t.shape[-1] == 4:
            if self.use_4D_normalizer:
                offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
                sampling_locations = reference_points_t[:, :, None, :, None, :2] + sampling_offsets / offset_normalizer[None, None, None, :, None, :] * reference_points_t[:, :, None, :, None, 2:] * 0.5
            else:
                sampling_locations = reference_points_t[:, :, None, :, None, :2] + sampling_offsets / self.n_points * reference_points_t[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError('Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points_t.shape[-1]))

        # prepare value list for core; if grouped needed reshape channels
        new_value = []
        if self.num_groups > 1:
            d_per_group = d_per_head // self.num_groups
            for lid_, (H_, W_) in enumerate(input_spatial_shapes):
                v = value[lid_]
                v = v.view(N, self.n_heads, d_per_head, H_, W_)
                v = v.view(N, self.n_heads, self.num_groups, d_per_group, H_, W_)
                v = v.permute(0, 1, 2, 3, 4, 5).reshape(N, self.n_heads * self.num_groups, d_per_group, H_, W_)
                v = v.reshape(N * (self.n_heads * self.num_groups), d_per_group, H_ * W_)
                new_value.append(v)
        else:
            new_value = value

        local_out = ms_deform_attn_core_pytorch(
            new_value, input_spatial_shapes, sampling_locations, attention_weights,
            sampling_modulation=sampling_modulation, region_kernel_size=(self.region_kernel_size if self.use_region_sampling else 1),
            is_energy=self.is_energy
        )  # (N, Len_q, d_model)

        # ---------- Grid attention branch (optional) ----------
        grid_out = None
        if self.use_grid_attention:
            # Build base grid per-level and cache (on device)
            # base_grid: (n_levels, P_grid, 2)
            if not hasattr(self, '_cached_base_grid') or self._cached_base_grid is None or self._cached_base_grid.device != query.device:
                base_grid = self._build_base_grid(input_spatial_shapes, device=query.device)  # (n_levels, P_grid, 2)
                self._cached_base_grid = base_grid
            else:
                base_grid = self._cached_base_grid  # (n_levels, P_grid, 2)

            P_grid = self.grid_num_points  # per-level grid points (gs*gs)

            # compute grid attention weights
            grid_att_w = self.grid_attention_weights(conditioning).view(N, Len_q, self.n_heads, self.n_levels * P_grid)
            grid_att_w = F.softmax(grid_att_w, -1).view(N, Len_q, self.n_heads, self.n_levels, P_grid)
            if self.num_groups > 1:
                grid_att_w = grid_att_w.unsqueeze(3).repeat(1, 1, 1, self.num_groups, 1, 1)
                grid_att_w = grid_att_w.view(N, Len_q, self.n_heads * self.num_groups, self.n_levels, P_grid)

            # prepare grid sampling locations: start from base grid and optionally add predicted offsets
            # base_grid shape (n_levels, P_grid, 2) -> expand to (N, Len_q, Mprime, n_levels, P_grid, 2)
            base_grid_exp = base_grid.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1,1,1,n_levels,P_grid,2)
            Mprime = self.n_heads * self.num_groups
            base_grid_exp = base_grid_exp.expand(N, Len_q, Mprime, -1, -1, -1).to(query.device)

            if self.use_grid_offsets:
                grid_offsets = self.grid_offsets(conditioning)
                if self.num_groups > 1:
                    grid_offsets = grid_offsets.view(N, Len_q, self.n_heads, self.n_levels, P_grid, self.num_groups, 2)
                    grid_offsets = grid_offsets.permute(0, 1, 2, 5, 3, 4, 6).reshape(N, Len_q, self.n_heads * self.num_groups, self.n_levels, P_grid, 2)
                else:
                    grid_offsets = grid_offsets.view(N, Len_q, self.n_heads, self.n_levels, P_grid, 2)
                # offsets need normalization similar to sampling_offsets: divide by offset_normalizer (flip HW)
                if reference_points_t.shape[-1] == 2:
                    offset_normalizer = torch.tensor(input_spatial_shapes, device=query.device)
                    offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.n_levels, 1, 2)
                    grid_sampling_locations = base_grid_exp + grid_offsets / offset_normalizer
                else:
                    # if 4D normalized, scale similarly to previous branch
                    if self.use_4D_normalizer:
                        offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
                        grid_sampling_locations = base_grid_exp + grid_offsets / offset_normalizer[None, None, None, :, None, :] * reference_points_t[:, :, None, :, None, 2:] * 0.5
                    else:
                        grid_sampling_locations = base_grid_exp + grid_offsets / (P_grid) * reference_points_t[:, :, None, :, None, 2:] * 0.5
            else:
                # no offsets: grid unchanged for all queries
                grid_sampling_locations = base_grid_exp  # normalized coords in [0,1]

            # Now sample with the same new_value list (reshaped for groups if needed)
            grid_out = ms_deform_attn_core_pytorch(
                new_value, input_spatial_shapes, grid_sampling_locations, grid_att_w,
                sampling_modulation=None, region_kernel_size=(self.region_kernel_size if self.use_region_sampling else 1),
                is_energy=self.is_energy,
            )  # (N, Len_q, d_model)

        # ---------- Fuse branches ----------
        if self.use_grid_attention and grid_out is not None:
            if self.use_grid_fusion and self.grid_fuse_proj is not None:
                # concat and linearly project
                fused = self.grid_fuse_proj(torch.cat([local_out, grid_out], dim=-1))
                return fused
            else:
                # simple sum
                return local_out + grid_out
        else:
            # only local branch active (original behaviour) -> return as before
            return local_out
