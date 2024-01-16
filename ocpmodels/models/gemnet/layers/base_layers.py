"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
import torch.nn as nn
from torch import Tensor
from typing import Callable, Union, Optional,Tuple
import torch
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_
from torch.nn.init import zeros_

#from ..initializers import he_orthogonal_init


# class Dense(torch.nn.Module):
#     """
#     Combines dense layer with scaling for swish activation.

#     Parameters
#     ----------
#         units: int
#             Output embedding size.
#         activation: str
#             Name of the activation function to use.
#         bias: bool
#             True if use bias.
#     """

#     def __init__(self, in_features, out_features, bias=False, activation=None):
#         super().__init__()

#         self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
#         self.reset_parameters()

#         if isinstance(activation, str):
#             activation = activation.lower()
#         if activation in ["swish", "silu"]:
#             self._activation = ScaledSiLU()
#         elif activation == "siqu":
#             self._activation = SiQU()
#         elif activation is None:
#             self._activation = torch.nn.Identity()
#         else:
#             raise NotImplementedError(
#                 "Activation function not implemented for GemNet (yet)."
#             )

#     def reset_parameters(self, initializer=he_orthogonal_init):
#         initializer(self.linear.weight)
#         if self.linear.bias is not None:
#             self.linear.bias.data.fill_(0)

#     def forward(self, x):
#         x = self.linear(x)
#         x = self._activation(x)
#         return x




class DenseLayer(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module] = None,
        weight_init: Callable = kaiming_uniform_,
        bias_init: Callable = zeros_,
    ):
        self.weight_init = weight_init
        self.bias_init = bias_init
        super(DenseLayer, self).__init__(in_features, out_features, bias)

        if isinstance(activation, str):
            activation = activation.lower()
        if activation in ["swish", "silu"]:
            self._activation = ScaledSiLU()
        elif activation == "siqu":
            self._activation = SiQU()
        elif activation is None:
            self._activation = torch.nn.Identity()
        else:
            raise NotImplementedError(
                "Activation function not implemented for GemNet (yet).")

    def reset_parameters(self):
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L106
        self.weight_init(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, input: torch.Tensor):
        y = F.linear(input, self.weight, self.bias)
        y = self._activation(y)
        return y



class ScaledSiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = torch.nn.SiLU()

    def forward(self, x):
        return self._activation(x) * self.scale_factor


class SiQU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._activation = torch.nn.SiLU()

    def forward(self, x):
        return x * self._activation(x)


class ResidualLayer(torch.nn.Module):
    """
    Residual block with output scaled by 1/sqrt(2).

    Parameters
    ----------
        units: int
            Output embedding size.
        nLayers: int
            Number of dense layers.
        layer_kwargs: str
            Keyword arguments for initializing the layers.
    """

    def __init__(
        self, units: int, nLayers: int = 2, layer=DenseLayer, **layer_kwargs
    ):
        super().__init__()
        self.dense_mlp = torch.nn.Sequential(
            *[
                layer(
                    in_features=units,
                    out_features=units,
                    bias=False,
                    **layer_kwargs
                )
                for _ in range(nLayers)
            ]
        )
        self.inv_sqrt_2 = 1 / math.sqrt(2)

    def forward(self, input):
        x = self.dense_mlp(input)
        x = input + x
        x = x * self.inv_sqrt_2
        return x
