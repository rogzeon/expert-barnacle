from torch import nn
import torch
from dataclasses import dataclass, asdict


# figure out what this actually does later
# TODO: Remove this comment
class FourierFeatures(nn.Module):
    def __init__(self, in_dim, num_features, scale=10.0):
        super().__init__()
        self.B = nn.Parameter(
            torch.randn(in_dim, num_features) * scale, requires_grad=False
        )

    def forward(self, x):
        x_proj = 2 * torch.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# Simple NN for now.
class FCN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        fourier_features=64,
        fourier_scale=10.0,
    ):
        super().__init__()
        # self.fourier = FourierFeatures(input_dim, fourier_features, scale=fourier_scale)
        # fourier_dim = fourier_features * 2  # sin + cos
        activation = nn.SiLU
        # self.fcs = nn.Sequential(*[nn.Linear(fourier_dim, hidden_dim), activation()])
        self.fcs = nn.Sequential(*[nn.Linear(input_dim, hidden_dim), activation()])
        self.fch = nn.Sequential(
            *[
                nn.Sequential(*[nn.Linear(hidden_dim, hidden_dim), activation()])
                for _ in range(num_layers - 1)
            ]
        )
        self.fce = nn.Linear(hidden_dim, output_dim)
        self.metadata = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
        }

    def forward(self, x):
        # x = self.fourier(x)
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


@dataclass
class Domain:
    """
    For convenient & flexible storage of the PDE domain.

    dims: stores the minimum & maximum values per dimension,
    as well as the resolution in that dimension

    t_min: initial time value
    t_max: final time value
    _t_res: time resolution
    """

    dims: list[tuple[float, float, float]]
    t_min: float
    t_max: float
    _t_res: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Domain":
        return cls(
            dims=[tuple(pair) for pair in d["dims"]],
            t_min=d["t_min"],
            t_max=d["t_max"],
            _t_res=d["_t_res"],
        )

    @property
    def nt(self) -> int:
        return int(self._t_res * (self.t_max - self.t_min))

    def ndim(self, dim) -> int:
        return int(self.dims[dim][2] * (self.dims[dim][1] - self.dims[dim][0]))

    def mindim(self, dim):
        return self.dims[dim][0]

    def maxdim(self, dim):
        return self.dims[dim][1]
