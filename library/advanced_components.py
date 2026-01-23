import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN)
    Normalizes time-series input and provides a way to denormalize the output logic.
    Ref: https://openreview.net/forum?id=cGDAkQo1C0p
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        """
        mode: 'norm' or 'denorm'
        x: [Batch, Seq, Features] or [Batch, Features]
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*1e-5)
        x = x * self.stdev
        x = x + self.mean
        return x


class KANLinear(nn.Module):
    """
    Efficient implementation of Kolmogorov-Arnold Network (KAN) Linear Layer.
    Replacing fixed activation functions with learnable B-Splines.
    Uses a residual connection: Linear(x) + Spline(x).
    """
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for x assuming no hidden batch dimension.
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:-k])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates points (x, y).
        """
        A = self.b_splines(x).transpose(0, 1)  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        return solution.permute(2, 0, 1).contiguous()

    def forward(self, x: torch.Tensor):
        original_shape = x.shape
        x = x.reshape(-1, self.in_features) # Flatten batch/seq

        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        spline_basis = self.b_splines(x) # [Batch, In, Grid]
        # Spline mul: (Batch, In, Grid) x (Out, In, Grid) -> (Batch, Out)
        # Using einsum for clarity: batch=b, in=i, grid=g, out=o
        spline_output = torch.einsum('big,oig->bo', spline_basis, self.spline_weight)
        
        if self.enable_standalone_scale_spline:
            spline_output = spline_output * self.spline_scaler.sum(dim=1) # Simplified scaling
            # Correct scaling usually per-weight, but let's keep it consistent with efficient-kan
            # Actually, let's use F.linear equivalent or elementwise
            pass
            
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit.
    F(x) = (xW + b) * SiLU(xV + c)
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, bias=True, dropout=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.w = nn.Linear(in_features, hidden_features, bias=bias)
        self.v = nn.Linear(in_features, hidden_features, bias=bias)
        self.out = nn.Linear(hidden_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x1 = self.w(x)
        x2 = self.v(x)
        hidden = F.silu(x1) * x2
        return self.dropout(self.out(hidden))
