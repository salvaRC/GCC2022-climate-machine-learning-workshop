import math
from typing import Sequence, Optional, Dict, Union

import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig

from src.models.base_model import BaseModel
from src.models.modules.mlp_module import MLP_module


class MLP(BaseModel):
    r"""Multi-layer perceptron (MLP) model.
        This model is agnostic to any spatial structure in the data, since it operates on 1D data vectors
        (spatial dimensions are flattened to 1D).

    Args:
        hidden_dims (List[int]): The hidden dimensions of the MLP (e.g. [100, 100, 100])
        datamodule_config (DictConfig): The config of the datamodule (e.g. produced by hydra <config>.yaml file parsing)
        net_normalization (str): One of ['batch_norm', 'layer_norm', 'none']. Default: "none"
        activation_function (str): The activation function of the MLP. Default: 'gelu'
        dropout (float): How much dropout to use in the MLP. Default: 0.0 (no dropout)
        residual (bool): Whether to use residual connections in the MLP. Default: ``False``
        residual_learnable_lam (bool): Whether to use residual connections with learnable lambdas
        output_activation_function (str, bool, optional): By default no output activation function is used (None).
            If a string is passed, is must be the name of the desired output activation (e.g. 'softmax')
            If True, the same activation function is used as defined by the arg `activation_function`.
    """

    def __init__(self,
                 hidden_dims: Sequence[int],
                 datamodule_config: DictConfig = None,
                 net_normalization: Optional[str] = None,
                 activation_function: str = 'gelu',
                 dropout: float = 0.0,
                 residual: bool = False,
                 output_activation_function: Optional[Union[str, bool]] = None,
                 *args, **kwargs):
        super().__init__(datamodule_config=datamodule_config, *args, **kwargs)
        # The following saves all the args that are passed to the constructor to self.hparams
        #   e.g. access them with self.hparams.hidden_dims
        self.save_hyperparameters()

        if isinstance(self.spatial_dims, int):
            mlp_total_spatial_dims = self.spatial_dim
            self.output_tensor_shape = (-1, self.num_output_channels, self.spatial_dim)
        else:
            mlp_total_spatial_dims = math.prod(self.spatial_dims)
            self.output_tensor_shape = tuple([-1, self.num_output_channels] + list(self.spatial_dims))

        self.input_dim = self.num_input_channels * mlp_total_spatial_dims
        self.output_dim = self.num_output_channels * mlp_total_spatial_dims

        self.example_input_array = torch.randn(1, self.input_dim)
        self.num_layers = len(hidden_dims)

        self.mlp = MLP_module(
            self.input_dim, hidden_dims, self.output_dim,
            net_normalization=net_normalization,
            activation_function=activation_function, dropout=dropout,
            residual=residual,
            output_normalization=False,
            output_activation_function=output_activation_function
        )

    def forward(self, X: Tensor) -> Tensor:
        r"""Forward the input through the MLP.

        Shapes:
            - Input: :math:`(B, C_{in}, *)`
            - Output: :math:`(B, C_{out}, *)`,

            where :math:`B` is the batch size, :math:`*` is the spatial dimension(s) of the data,
            and :math:`C_{in}` (:math:`C_{out}`) is the number of input (output) features.
        """
        flattened_X = torch.flatten(X, start_dim=1)
        flattened_Y = self.mlp(flattened_X)
        # Reshape back into spatially structured outputs
        Y = flattened_Y.view(self.output_tensor_shape)
        return Y
