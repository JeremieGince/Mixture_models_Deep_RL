import numpy as np
import torch

from Models.short_memory_model import SMModel
from typing import Tuple, List, Union, Iterable


class SMGRU(SMModel):
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
            self.linear_block(np.prod(in_shape), hidden_dim),
            *[self.linear_block(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)]
        ])

        self.gru = torch.nn.GRU(hidden_dim, hidden_dim)

        self.q_predictor = torch.nn.Sequential(*[
            torch.nn.Linear(memory_size * hidden_dim, int(np.prod(out_shape)))
        ])

    def forward(self, *inputs):
        [state, ] = inputs
        state_features = self.backbone(state.float())

        lstm_output, _ = self.gru(state_features.permute(1, 0, 2))
        env_features = torch.flatten(lstm_output.permute(1, 0, 2), start_dim=1)
        q_values = self.q_predictor(env_features)
        return q_values
