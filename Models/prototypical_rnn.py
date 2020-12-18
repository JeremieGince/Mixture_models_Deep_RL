import numpy as np
import torch
from torch import nn

from Models.short_memory_model import SMModel
from typing import Tuple, List, Union, Iterable


class ProtoSMRNN(SMModel):
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

        self.state_backbone = torch.nn.Sequential(*[
            nn.Flatten(),
            self.linear_block(np.prod(in_shape), hidden_dim),
            *[self.linear_block(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)]
        ])

        self.context_backbone = torch.nn.Sequential(*[
            nn.Flatten(),
            self.linear_block(np.prod(in_shape), hidden_dim),
            *[self.linear_block(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)]
        ])

        self.fusion_layer = torch.nn.Sequential(*[
            self.linear_block(2 * hidden_dim, hidden_dim)
        ])

        self.q_predictor = torch.nn.Sequential(*[
            torch.nn.Linear(hidden_dim, int(np.prod(out_shape)))
        ])

    def forward(self, *inputs):
        [state, ] = inputs
        curr_state = state[:, -1]
        proto_context = torch.mean(state[:, :-1], dim=1)

        state_features = self.state_backbone(curr_state.float())
        context_features = self.context_backbone(proto_context.float())
        fusion_state = torch.cat([state_features, context_features], dim=-1)
        fusion_features = self.fusion_layer(fusion_state)
        q_values = self.q_predictor(fusion_features)
        return q_values
