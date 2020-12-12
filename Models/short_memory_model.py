from typing import Tuple, List, Union, Iterable

import torch


class SMModel(torch.nn.Module):
    """
    """

    def __init__(self,
                 in_shape: Union[Tuple, List, Iterable],
                 out_shape: Union[Tuple, List, Iterable],
                 n_hidden_layers: int = 3,
                 hidden_dim: int = 64,
                 memory_size: int = 10,
                 **kwargs):
        super().__init__()
        self.in_dim = in_shape
        self.out_dim = out_shape
        self.n_hidden_layers = n_hidden_layers
        self.hidden_dim = hidden_dim
        self.memory_size: int = memory_size
        self.kwargs = kwargs
        self.c0 = torch.zeros(size=(1, hidden_dim,))
        self.h0 = torch.zeros(size=(1, hidden_dim,))

    def forward(self, *inputs):
        raise NotImplementedError()