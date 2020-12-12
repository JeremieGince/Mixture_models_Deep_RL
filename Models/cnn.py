from typing import Tuple, List, Union, Iterable

import numpy as np
import torch
from torch import nn

from Models.short_memory_model import SMModel
from Modules.ConvNd import ConvNd


class SMCNN(SMModel):
    """

    """

    def __init__(self,
                 in_shape: Union[Tuple, List, Iterable],
                 out_shape: Union[Tuple, List, Iterable],
                 n_hidden_layers: int = 1,
                 hidden_dim: int = 64,
                 memory_size: int = 10,
                 **kwargs):
        n_dim = len(in_shape)
        self.n_dim = n_dim
        super().__init__(
            in_shape,
            out_shape,
            n_hidden_layers,
            hidden_dim,
            memory_size,
            **kwargs,
        )
        stride = 1
        self.permute = kwargs.get("permute", False)

        self.linear_block = lambda i, o: torch.nn.Sequential(*[
            torch.nn.Linear(i, o),
            torch.nn.ReLU(),
        ])

        self.conv1d_bloc = lambda c_i, c_o, k: torch.nn.Sequential(
            nn.Conv1d(in_channels=c_i, out_channels=c_o, kernel_size=k, stride=stride, bias=True),
            nn.BatchNorm1d(c_o),
            nn.ReLU(),
        )
        self.conv2d_bloc = lambda c_i, c_o, k: torch.nn.Sequential(
            nn.Conv2d(in_channels=c_i, out_channels=c_o, kernel_size=k, stride=stride, bias=True),
            nn.BatchNorm2d(c_o),
            nn.ReLU(),
        )
        self.conv3d_bloc = lambda c_i, c_o, k: torch.nn.Sequential(
            nn.Conv3d(in_channels=c_i, out_channels=c_o, kernel_size=k, stride=stride, bias=True),
            nn.BatchNorm3d(c_o),
            nn.ReLU(),
        )
        self.convNd_bloc = lambda c_i, c_o, k: torch.nn.Sequential(
            ConvNd(in_channels=c_i, out_channels=c_o, num_dims=n_dim,
                   kernel_size=k, stride=stride, padding=0, use_bias=True),
            nn.SyncBatchNorm(c_o),
            nn.ReLU(),
        )

        self.conv_func_ndim = {
            1: self.conv1d_bloc,
            2: self.conv2d_bloc,
            3: self.conv3d_bloc,
        }

        self.conv_bloc_gen = self.conv_func_ndim[n_dim] if n_dim in self.conv_func_ndim else self.convNd_bloc

        if self.permute:
            self.complete_in_shape = (1, *in_shape, memory_size)

            self.sh_idx = list(range(len(self.complete_in_shape)))
            c = self.sh_idx.pop(1)
            self.sh_idx.append(c)
        else:
            self.complete_in_shape = (1, memory_size, *in_shape)

        k_list = list(reversed(range(0, max(0, self.complete_in_shape[-1] // 2))))
        layers = [
            self.conv_bloc_gen(self.complete_in_shape[1], hidden_dim, k_list[0]),
        ]
        for i, k in enumerate(k_list):
            if k > 0 and i < n_hidden_layers:
                layers.append(self.conv_bloc_gen(hidden_dim, hidden_dim, k))

        self.backbone = torch.nn.Sequential(*layers)

        # _in = torch.ones(self.complete_in_shape)
        # for layer in layers:
        #     print(f"_in.shape: {_in.shape}")
        #     print(f"layer: {layer}")
        #     _out = layer(_in)
        #     print(f"_out.shape: {_out.shape}")
        #     _in = _out

        self.backbone_output_shape: tuple = tuple(self.backbone(torch.ones(self.complete_in_shape)).shape)
        hh_dim: int = int(np.prod(self.backbone_output_shape))
        self.q_predictor = torch.nn.Sequential(*[
            nn.Flatten(),
            self.linear_block(hh_dim, hh_dim),
            torch.nn.Linear(hh_dim, int(np.prod(out_shape)))
        ])

    def forward(self, *inputs):
        [state, ] = inputs
        if self.permute:
            state = state.permute(self.sh_idx)
        state_features = self.backbone(state.float())
        q_values = self.q_predictor(state_features)
        return q_values
