import numpy as np
import torch
from torch import nn

from Models.short_memory_model import SMModel
from Modules.layers import QuantumPseudoLinearLayer
from typing import Tuple, List, Union, Iterable


class SMQNNModel(SMModel):
    """
    Quantum Neural Network
    """

    def __init__(self,
                 in_shape: Union[Tuple, List, Iterable],
                 out_shape: Union[Tuple, List, Iterable],
                 n_hidden_layers: int = 2,
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
        self.complete_in_shape = (1, memory_size, *in_shape)
        self.nb_qubits = kwargs.get("nb_qubits", 2)
        kwargs["nb_qubits"] = self.nb_qubits
        self.linear_block = lambda i, o: torch.nn.Sequential(*[
            torch.nn.Linear(i, o),
            torch.nn.ReLU(),
        ])

        self.q_backbone = torch.nn.Sequential(*[
            nn.Flatten(),
            self.linear_block(memory_size * np.prod(in_shape), self.nb_qubits),
            *[QuantumPseudoLinearLayer(**kwargs)
              for _ in range(kwargs.get("nb_q_layer", 2))],
        ])
        self.q_backbone_output_shape: tuple = tuple(self.q_backbone(torch.ones(self.complete_in_shape)).shape)
        q_hh_dim: int = int(np.prod(self.q_backbone_output_shape))

        self.c_backbone = torch.nn.Sequential(*[
            nn.Flatten(),
            self.linear_block(memory_size * np.prod(in_shape), hidden_dim),
            *[self.linear_block(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)]
        ])

        self.fusion_layer = torch.nn.Sequential(*[
            self.linear_block(hidden_dim + q_hh_dim, hidden_dim)
        ])

        self.q_predictor = torch.nn.Sequential(*[
            nn.Flatten(),
            torch.nn.Linear(hidden_dim, int(np.prod(out_shape)))
        ])

    def forward(self, *inputs):
        [state, ] = inputs
        q_features = self.q_backbone(state.float())
        c_features = self.c_backbone(state.float())
        fusion_features = torch.cat([q_features, c_features], dim=-1)
        fusion_features = self.fusion_layer(fusion_features)
        q_values = self.q_predictor(fusion_features)
        return q_values
