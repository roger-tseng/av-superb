import torch
from torch import nn
import torch.nn.functional as F

from models.expRNN.orthogonal import Orthogonal
from models.expRNN.initialization import henaff_init_, cayley_init_
from models.expRNN.trivializations import cayley_map, expm


class FactorLinear(nn.Module):
    def __init__(
        self,
        num_modality_groups,
        input_size,
        orthogonal_size,
        output_size,
        mode,
        init,
        K="100",
    ):
        super(FactorLinear, self).__init__()
        for i in range(num_modality_groups):
            self.add_module(
                "ms{}_linear".format(i),
                nn.Linear(input_size, orthogonal_size, bias=False)
            )

        if init == "cayley":
            _init = cayley_init_
        elif init == "henaff":
            _init = henaff_init_

        if K != "infty":
            K = int(K)

        if mode == "static":
            _mode = "static"
            param = expm
        elif mode == "dtriv":
            _mode = ("dynamic", K, 100)
            param = expm
        elif mode == "cayley":
            _mode = "static"
            param = cayley_map

        for i in range(num_modality_groups):
            self.add_module(
                "orthogonal_ms{}_linear".format(i),
                Orthogonal(
                    orthogonal_size,
                    orthogonal_size,
                    _init,
                    _mode,
                    param,
                )
            )

        self.s_linear = nn.Linear(orthogonal_size, output_size)

    def forward(self, x, modality_idx):
        x = getattr(self, "ms{}_linear".format(modality_idx))(x)
        x = F.normalize(x, dim=-1)
        x = getattr(self, "orthogonal_ms{}_linear".format(modality_idx))(x)
        x = self.s_linear(x)
        return x
