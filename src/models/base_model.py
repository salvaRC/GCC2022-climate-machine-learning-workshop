import logging
import time
from typing import Optional, List, Any, Dict, Sequence, Union, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

from src.utilities.utils import get_logger, get_loss, raise_error_if_invalid_value


class BaseModel(pl.LightningModule):
    r""" This is a template base class, that should be inherited by any stand-alone ML model.
    Methods that need to be implemented by your concrete ML model (just as if you would define a :class:`torch.nn.Module`):
        - :func:`__init__`
        - :func:`forward`

    The other methods may be overridden as needed.
    It is recommended to define the attribute
        >>> self.example_input_array = torch.randn(<YourModelInputShape>)  # batch dimension can be anything, e.g. 7


    .. note::
        Please use the function :func:`predict` at inference time for a given input tensor, as it postprocesses the
        raw predictions from the function :func:`raw_predict` (or model.forward or model())!

    Args:
        datamodule_config: DictConfig with the configuration of the datamodule
        loss_function (str): The name of the loss function. Default: 'mean_squared_error'
        name (str): optional string with a name for the model
        verbose (bool): Whether to print/log or not

    Read the docs regarding LightningModule for more information:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(self,
                 datamodule_config: DictConfig = None,
                 learning_rate: float = 3e-5,
                 loss_function: str = 'mean_squared_error',
                 name: str = "",
                 verbose: bool = True,
                 ):
        super().__init__()
        # The following saves all the args that are passed to the constructor to self.hparams
        #   e.g. access them with self.hparams.learning_rate
        self.save_hyperparameters()
        # Get a logger
        self.log_text = get_logger(name=self.__class__.__name__ if name == '' else name)
        self.name = name
        self.verbose = verbose
        if not self.verbose:  # turn off info level logging
            self.log_text.setLevel(logging.WARN)

        # Data dimensions - Todo: infer them from the datamodule config
        self._data_dir = datamodule_config.get('data_dir', None) if datamodule_config is not None else None
        self.spatial_dims = (60, 60)  # (lat, lon), assumes the OISSTv2 tiled data
        self.num_input_channels = 1
        self.num_output_channels = 1

        # The loss function. The mean squared error is the usual go-to for regression
        self._loss_function = get_loss(loss_function)  # e.g. nn.MSELoss()

        # Timing variables to track the training/epoch/validation time
        self._start_validation_epoch_time = self._start_test_epoch_time = self._start_epoch_time = None

        # Metrics
        self.val_metrics = nn.ModuleDict({'val/mae': torchmetrics.MeanAbsoluteError()})
        self.test_metrics = nn.ModuleDict({'test/mae': torchmetrics.MeanAbsoluteError()})

        # Check that the args/hparams are valid
        self._check_args()

    @property
    def n_params(self):
        """ Returns the number of parameters in the model """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def data_dir(self) -> str:
        if self._data_dir is None:
            self._data_dir = self.trainer.datamodule.hparams.data_dir
        return self._data_dir

    def _check_args(self):
        """Check if the arguments are valid."""
        pass

    def forward(self, X: Tensor):
        r""" Standard ML model forward pass (to be implemented by the specific ML model).

        Args:
            X (Tensor): Input data tensor of shape :math:`(B, *, C_{in})`
        Shapes:
            - Input: :math:`(B, *, C_{in})`,

            where :math:`B` is the batch size, :math:`*` is the spatial dimension(s) of the data,
            and :math:`C_{in}` is the number of input features/channels.
        """
        raise NotImplementedError('Base model is an abstract class!')

    # --------------------- training with PyTorch Lightning
    def on_train_start(self) -> None:
        """ Log some info about the model/data at the start of training """
        self.log('Parameter count', float(self.n_params))
        self.log('Training set size', float(len(self.trainer.datamodule._data_train)))
        self.log('Validation set size', float(len(self.trainer.datamodule._data_val)))

    def on_train_epoch_start(self) -> None:
        self._start_epoch_time = time.time()

    def training_step(self, batch: Any, batch_idx: int):
        """ One step of training (Backpropagation is done on the loss returned at the end of this function) """
        x, y = batch   # x = inputs, y = targets (or ground truth)
        y_hat = self.model(x)  # y_hat = model predictions
        loss = self._loss_function(y_hat, y)
        # Logging of train loss and other diagnostics
        train_log = {
            "train/loss": loss.item(),
            'n_zero_gradients': sum(
            [int(torch.count_nonzero(p.grad == 0))
             for p in self.parameters() if p.grad is not None
             ]) / self.n_params}

        # Count number of zero gradients as diagnostic tool

        self.log_dict(train_log, prog_bar=False)
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        train_time = time.time() - self._start_epoch_time
        self.log_dict({'epoch': float(self.current_epoch), "time/train": train_time})

    # --------------------- evaluation with PyTorch Lightning
    def evaluation_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                        batch_idx: int,
                        torch_metrics: nn.ModuleDict,
                        split_name: str,  # 'val' or 'test'
                        **kwargs
                        ) -> Dict[str, float]:
        """ Defines a single evaluation loop/iteration over one batch of data. """
        x, y = batch
        y_hat = self(x)  # predict with the model (self(x) is same as self.forward(x))
        log_dict = {f'{split_name}/loss': self._loss_function(y_hat, y)}
        # Compute metrics
        for metric_name, metric in torch_metrics.items():
            # The two following lines need to be separate!
            metric(y_hat, y)
            log_dict[metric_name] = metric
        self.log_dict(log_dict, on_step=False, on_epoch=True, **kwargs)  # log metric objects
        return log_dict

    def on_validation_epoch_start(self) -> None:
        self._start_validation_epoch_time = time.time()

    # Here, we will use the same evaluation loop for validation and testing
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, float]:
        return self.evaluation_step(batch, batch_idx, split_name='val', torch_metrics=self.val_metrics, prog_bar=True)

    def validation_epoch_end(self, outputs: List[Any]) -> dict:
        val_time = time.time() - self._start_validation_epoch_time
        self.log_dict({'time/val': val_time})


    # ---------------------------------------------------------------------- Optimizers and scheduler(s)
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """ Define which optimization algorithm to use """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    # Unimportant methods
    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
