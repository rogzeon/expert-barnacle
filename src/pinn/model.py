import torch
from torch import nn


# Simple NN for now.
class FCN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
    ):
        super().__init__()
        activation = nn.SiLU
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
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


class ConstrainedFCN(nn.Module):
    """
    FCN with ansatz. Helpful for dealing with non-differentiable
    boundary conditions.

    The ansatz is

        u(x, t) = w(t) * payoff(x) + (1 - w(t)) * base_net(x, t)

    and can be specified by overriding ``payoff()`` & ``forward()``.

    As a consequence, u(x, t_max) = payoff(x), hardcoding the IC.

    Since w(t) is gaussian, as the network gets further from the
    pathological value, the hardcoded payoff has less influence
    on the network's predictions.

    ``kink_width`` determines how fast w(t) falls off.

    IMPORTANT NOTE: Currently incompatible with eval scripts.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        strike,
        t_max,
        kink_width,
    ):
        super().__init__()
        self.base_net = FCN(input_dim, output_dim, hidden_dim, num_layers)
        self.strike = strike
        self.t_max = t_max
        self.kink_width = kink_width
        self.metadata = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "strike": strike,
            "t_max": t_max,
            "kink_width": kink_width,
        }

    def payoff(self, x):
        return torch.clamp(x - self.strike, min=0.0)

    def weight(self, t):
        return torch.exp(-(((self.t_max - t) / self.kink_width) ** 2))

    def forward(self, xt):
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        w = self.weight(t)
        return w * self.payoff(x) + (1 - w) * self.base_net(xt)
