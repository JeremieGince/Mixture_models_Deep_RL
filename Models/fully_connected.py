from typing import Tuple, List, Union, Iterable

import numpy as np
import torch
from torch import nn

from Models.short_memory_model import SMModel


class SMNNModel(SMModel):
    """
    """

    def __init__(self,
                 in_shape: Union[Tuple, List, Iterable],
                 out_shape: Union[Tuple, List, Iterable],
                 n_hidden_layers: int = 3,
                 hidden_dim: int = 64,
                 memory_size: int = 10,
                 **kwargs):
        super().__init__(
            in_shape,
            out_shape,
            n_hidden_layers,
            hidden_dim,
            memory_size,
            **kwargs,
        )

        self.linear_block = lambda i, o: torch.nn.Sequential(*[
            torch.nn.Linear(i, o),
            torch.nn.ReLU(),
        ])

        self.backbone = torch.nn.Sequential(*[
            nn.Flatten(),
            self.linear_block(memory_size*np.prod(in_shape), hidden_dim),
            *[self.linear_block(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)]
        ])

        self.q_predictor = torch.nn.Sequential(*[
            torch.nn.Linear(hidden_dim, int(np.prod(out_shape)))
        ])

    def forward(self, *inputs):
        [state, ] = inputs
        features = self.backbone(state.float())
        q_values = self.q_predictor(features)
        return q_values
