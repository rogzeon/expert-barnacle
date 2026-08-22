
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
