# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat
from timm.models.layers import trunc_normal_

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj
except ImportError:
    selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None, None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None



def direct_tokens(x, w=7, w_first=False):
    B, L, C = x.shape
    H = W = int(L ** 0.5)
    Hg = Wg = H // w
    if w_first:
        x = x.view(B, Hg, w, Wg, w, C).permute(0, 3, 1, 4, 2, 5).reshape(B, L, C)
    else:
        x = x.view(B, Hg, w, Wg, w, C).permute(0, 1, 3, 2, 4, 5).reshape(B, L, C)
    return x

def reverse_tokens(x, w=7, w_first=False):
    B, L, C = x.shape
    H = W = int(L ** 0.5)
    Hg = Wg = H // w
    if w_first:
        x = x.view(B, Wg, Hg, w, w, C).permute(0, 2, 4, 1, 3, 5).reshape(B, L, C)
    else:
        x = x.view(B, Hg, Wg, w, w, C).permute(0, 1, 3, 2, 4, 5).reshape(B, L, C)
    return x


class DynamicScan(nn.Module):
    def __init__(self, dim, hidden_dim=96, window_size=2):
        super().__init__()
        self.window_size = window_size
        self.num_tokens = window_size**2
        self.tokens = nn.Parameter(torch.zeros(1, 1, self.num_tokens, dim))

    def forward(self, x):
        B, L, D = x.shape
        x = x.view(B, -1, self.num_tokens, D)
        attn = self.tokens.expand(B, x.shape[1], -1, -1) @ x.transpose(-2, -1)  # [B, -1, N, N]
        # attn = F.gumbel_softmax(attn, hard=True)
        attn = attn.softmax(-1)
        new_x = (attn @ x).view(B, L, D)
        return attn, new_x

    def reverse(self, x, attn):
        B, L, D = x.shape
        x = x.view(B, -1, self.num_tokens, D)
        ori_x = attn.transpose(-2, -1) @ x
        return ori_x.view(B, L, D)


class MultiScan(nn.Module):

    CHOICES = ('h', 'h_flip', 'v', 'v_flip', 'w2', 'w2_flip', 'w7', 'w7_flip')

    def __init__(self, dim):
        super().__init__()
        self.choices = MultiScan.CHOICES
        self.norms = nn.ModuleList([nn.LayerNorm(dim, elementwise_affine=False) for _ in self.choices])
        self.weights = nn.Parameter(1e-3 * torch.randn(len(self.choices), 1, 1, 1))
        self._iter = 0

    def forward(self, xs):
        weights = self.weights.softmax(0)
        xs = [norm(x) for norm, x in zip(self.norms, xs)]
        xs = torch.stack(xs) * weights
        x = xs.sum(0)
        if self._iter % 200 == 0 and torch.distributed.get_rank() == 0:
            print(weights.detach().view(-1).tolist())
        self._iter += 1
        return x

    def multi_scan(self, x):
        """
        Input @x: shape [B, L, D]
        """
        xs = []
        for direction in self.choices:
            xs.append(self.scan(x, direction))
        return xs

    def multi_reverse(self, xs):
        new_xs = []
        for x, direction in zip(xs, self.choices):
            new_xs.append(self.reverse(x, direction))
        return new_xs

    def scan(self, x, direction='h'):
        """
        Input @x: shape [B, L, D]
        Return torch.Tensor: shape [B, L, D]
        """
        B, L, D = x.shape
        H = W = int(L ** 0.5)
        if direction == 'h':
            return x
        elif direction == 'h_flip':
            return x.flip([1])
        elif direction == 'v':
            return x.view(B, H, W, D).transpose(1, 2).reshape(B, L, D)
        elif direction == 'v_flip':
            return x.view(B, H, W, D).transpose(1, 2).reshape(B, L, D).flip([1])
        elif direction == 'w2':
            return direct_tokens(x, w=2, w_first=False)
        elif direction == 'w2_flip':
            return direct_tokens(x, w=2, w_first=False).flip([1])
        elif direction == 'w7':
            return direct_tokens(x, w=7, w_first=False)
        elif direction == 'w7_flip':
            return direct_tokens(x, w=7, w_first=False).flip([1])
        else:
            raise RuntimeError(f'Direction {direction} not found.')

    def reverse(self, x, direction='h'):
        """
        Input @x: shape [B, L, D]
        Return torch.Tensor: shape [B, L, D]
        """
        B, L, D = x.shape
        H = W = int(L ** 0.5)
        if direction == 'h':
            return x
        elif direction == 'h_flip':
            return x.flip([1])
        elif direction == 'v':
            return x.view(B, W, H, D).transpose(1, 2).reshape(B, L, D)
        elif direction == 'v_flip':
            return x.flip([1]).view(B, W, H, D).transpose(1, 2).reshape(B, L, D)
        elif direction == 'w2':
            return reverse_tokens(x, w=2, w_first=False)
        elif direction == 'w2_flip':
            return reverse_tokens(x.flip([1]), w=2, w_first=False)
        elif direction == 'w7':
            return reverse_tokens(x, w=7, w_first=False)
        elif direction == 'w7_flip':
            return reverse_tokens(x.flip([1]), w=7, w_first=False)
        else:
            raise RuntimeError(f'Direction {direction} not found.')    


class WindowScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, window_size=2, w_first=False):
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        ctx.shape = (B, C, H, W)
        ctx.window_size = window_size
        ctx.w_first = w_first
        
        # H = W = int(L ** 0.5)
        # Hg = Wg = H // w
        # if w_first:
        #     x = x.view(B, Wg, Hg, w, w, C).permute(0, 2, 4, 1, 3, 5).reshape(B, L, C)
        # else:
        #     x = x.view(B, Hg, Wg, w, w, C).permute(0, 1, 3, 2, 4, 5).reshape(B, L, C)
        return direct_tokens(x, window_size, w_first)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return reverse_tokens(grad, ctx.window_size, ctx.w_first), None, None
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)



class BiAttn(nn.Module):
    def __init__(self, in_channels, act_ratio=0.125, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = nn.LayerNorm(in_channels)
        self.global_reduce = nn.Linear(in_channels, reduce_channels)
        # self.local_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        # self.spatial_select = nn.Linear(reduce_channels * 2, 1)
        self.gate_fn = gate_fn()

    def forward(self, x):
        ori_x = x
        x = self.norm(x)
        x_global = x.mean(1, keepdim=True)
        x_global = self.act_fn(self.global_reduce(x_global))
        # x_local = self.act_fn(self.local_reduce(x))

        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)  # [B, 1, C]
        # s_attn = self.spatial_select(torch.cat([x_local, x_global.expand(-1, x.shape[1], -1)], dim=-1))
        # s_attn = self.gate_fn(s_attn)  # [B, N, 1]

        attn = c_attn #* s_attn  # [B, N, C]
        return ori_x * attn


is_first = True
class DynMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="none"
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # self.conv1d = nn.Conv1d(
        #     in_channels=self.d_inner,
        #     out_channels=self.d_inner,
        #     bias=conv_bias,
        #     kernel_size=d_conv,
        #     groups=self.d_inner,
        #     padding=d_conv - 1,
        #     **factory_kwargs,
        # )

        self.activation = "silu"
        self.act = nn.SiLU()

        # self.x_proj = nn.Linear(
        #     self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        # )
        # self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # # Initialize special dt projection to preserve variance at initialization
        # dt_init_std = self.dt_rank**-0.5 * dt_scale
        # if dt_init == "constant":
        #     nn.init.constant_(self.dt_proj.weight, dt_init_std)
        # elif dt_init == "random":
        #     nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # else:
        #     raise NotImplementedError

        # # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        # dt = torch.exp(
        #     torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
        #     + math.log(dt_min)
        # ).clamp(min=dt_init_floor)
        # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        # inv_dt = dt + torch.log(-torch.expm1(-dt))
        # with torch.no_grad():
        #     self.dt_proj.bias.copy_(inv_dt)
        # # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # self.dt_proj.bias._no_reinit = True


        self.multi_scan = MultiScan(self.d_inner)
        '''new for search'''
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        for i in range(len(self.multi_scan.choices)):
            setattr(self, f'A_log_{i}', nn.Parameter(A_log))
            getattr(self, f'A_log_{i}')._no_weight_decay = True

            conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            setattr(self, f'conv1d_{i}', conv1d)

            x_proj = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            setattr(self, f'x_proj_{i}', x_proj)

            dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            # Initialize special dt projection to preserve variance at initialization
            dt_init_std = self.dt_rank**-0.5 * dt_scale
            if dt_init == "constant":
                nn.init.constant_(dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError

            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True

            setattr(self, f'dt_proj_{i}', dt_proj)

            D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            D._no_weight_decay = True
            setattr(self, f'D_{i}', D)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.attn = BiAttn(self.d_inner)

        return

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        assert bimamba_type == "v2"

        A_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
        self.A_b_log = nn.Parameter(A_b_log)
        self.A_b_log._no_weight_decay = True 

        self.conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_b = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_b._no_weight_decay = True



        '''c'''
        A_c = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_c_log = torch.log(A_c)  # Keep A_b_log in fp32
        self.A_c_log = nn.Parameter(A_c_log)
        self.A_c_log._no_weight_decay = True 

        self.conv1d_c = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_c = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_c = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.D_c = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_c._no_weight_decay = True


        '''d'''
        A_d = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_d_log = torch.log(A_d)  # Keep A_b_log in fp32
        self.A_d_log = nn.Parameter(A_d_log)
        self.A_d_log._no_weight_decay = True 

        self.conv1d_d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_d = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_d = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.D_d = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_d._no_weight_decay = True


        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.attn = BiAttn(self.d_inner)

        # self.dyn_scan_a = DynamicScan(self.d_inner * 2)
        # self.dyn_scan_b = DynamicScan(self.d_inner * 2)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out
 

        xz = self.in_proj(hidden_states)

        # A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            xs = self.multi_scan.multi_scan(xz)
            outs = []
            for i, xz in enumerate(xs):
                xz = rearrange(xz, "b l d -> b d l")
                A = -torch.exp(getattr(self, f'A_log_{i}').float())
                conv1d = getattr(self, f'conv1d_{i}')
                x_proj = getattr(self, f'x_proj_{i}')
                dt_proj = getattr(self, f'dt_proj_{i}')
                D = getattr(self, f'D_{i}')

                out = mamba_inner_fn_no_out_proj(
                    xz,
                    conv1d.weight,
                    conv1d.bias,
                    x_proj.weight,
                    dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    D.float(),
                    delta_bias=dt_proj.bias.float(),
                    delta_softplus=True,
                )
                outs.append(rearrange(out, "b d l -> b l d"))

            outs = self.multi_scan.multi_reverse(outs)
            outs = [self.attn(out) for out in outs]
            out = self.multi_scan(outs)
            out = F.linear(out, self.out_proj.weight, self.out_proj.bias)

            return out

            if self.bimamba_type == "v2":
                A_b = -torch.exp(self.A_b_log.float())
                A_c = -torch.exp(self.A_c_log.float())
                A_d = -torch.exp(self.A_d_log.float())
                
                xz_w2 = direct_tokens(xz, 2)
                # attn_a, xz_a = self.dyn_scan_a(xz_w2)
                xz_a = rearrange(xz_w2, "b l d -> b d l")
                # xz_b = rearrange(direct_tokens(xz, 2, True), "b l d -> b d l")
                # attn_b, xz_b = self.dyn_scan_b(xz_w2)
                # xz_b = rearrange(xz_b, "b l d -> b d l")

                xz_b = xz_a.flip([-1])
                xz_c = rearrange(xz, "b l d -> b d l")
                xz_d = xz_c.flip([-1])
                # xz_d = rearrange(direct_tokens(xz, 14, True), "b l d -> b d l")


                out = mamba_inner_fn_no_out_proj(
                    xz_a,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                # print(out.shape)
                out_b = mamba_inner_fn_no_out_proj(
                    xz_b,
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                out_c = mamba_inner_fn_no_out_proj(
                    xz_c,
                    self.conv1d_c.weight,
                    self.conv1d_c.bias,
                    self.x_proj_c.weight,
                    self.dt_proj_c.weight,
                    A_c,
                    None,
                    None,
                    self.D_c.float(),
                    delta_bias=self.dt_proj_c.bias.float(),
                    delta_softplus=True,
                )
                out_d = mamba_inner_fn_no_out_proj(
                    xz_d,
                    self.conv1d_d.weight,
                    self.conv1d_d.bias,
                    self.x_proj_d.weight,
                    self.dt_proj_d.weight,
                    A_d,
                    None,
                    None,
                    self.D_d.float(),
                    delta_bias=self.dt_proj_d.bias.float(),
                    delta_softplus=True,
                )

                # out = F.linear(rearrange(out + out_b, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
                # out = rearrange(out, "b d l -> b l 1 d") * rearrange(probs_a, "b l k -> b l k 1")  # [b l k d]
                # out = out.transpose(1, 2).sum(2)  # [b k d]
                # out_b = rearrange(out_b, "b d l -> b l 1 d") * rearrange(probs_b, "b l k -> b l k 1")  # [b l k d]
                # out_b = out_b.transpose(1, 2).sum(2)  # [b k d]
                # out = probs_a.transpose(-2, -1) @ rearrange(out, "b d l -> b l d")  # [b l d]
                # out_b = probs_b.transpose(-2, -1) @ rearrange(out_b, "b d l -> b l d")  # [b l d]
                out = rearrange(out, "b d l -> b l d")  # [b l d]
                out_b = rearrange(out_b.flip([-1]), "b d l -> b l d")  # [b l d]
                # out = self.dyn_scan_a.reverse(out, attn_a)
                out = reverse_tokens(out, 2)
                # out_b = self.dyn_scan_b.reverse(out_b, attn_b)
                out_b = reverse_tokens(out_b, 2)
                out_c = rearrange(out_c, "b d l -> b l d")  # [b l d]
                out_d = rearrange(out_d.flip([-1]), "b d l -> b l d")  # [b l d]

                out = self.attn(out)
                out_b = self.attn(out_b)
                out_c = self.attn(out_c)
                out_d = self.attn(out_d)

                # F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)
                # out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
                out = F.linear(out + out_b + out_c + out_d, self.out_proj.weight, self.out_proj.bias)
                # out = F.linear(out + out_b, self.out_proj.weight, self.out_proj.bias)
            else:
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                conv_state.copy_(x[:, :, -self.d_conv :])  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)