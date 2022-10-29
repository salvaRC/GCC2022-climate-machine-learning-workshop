import multiprocessing
import os
from typing import Optional
import logging
import numpy as np
import xarray as xr
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import dask

# Define the temporal slices that will be used for training, validating, and testing
# Important: training data should not contain data that is temporally near the test data (to remove autocorrelation issues)
TRAIN_SLICE = slice(None, '2018-12-31')
VAL_SLICE = slice('2019-01-01', '2019-12-31')
TEST_SLICE = slice('2020-01-01', '2021-12-31')


class OISSTv2DataModule(pl.LightningDataModule):
    """
    Data module for the OISSTv2 dataset of daily sea surface temperatures.
    A data module encapsulates the data loading, preprocessing, and data splits needed for training, validation, and testing a neural network model.
    These generated PyTorch-ready TensorDataset's are saved in the self._data_train, self._data_val, and self._data_test attributes when calling the setup() function.
    """
    _data_train: TensorDataset
    _data_val: TensorDataset
    _data_test: TensorDataset

    def __init__(self, data_dir: str, horizon: int = 1, batch_size: int = 32, eval_batch_size: int = 64, model_config=None):
        """
        Args:
            data_dir (str):  A path to the data folder that contains the input and output files.
            horizon (int): The number of time steps to predict into the future (e.g. 1 for 1 day ahead prediction).
            batch_size (int): Batch size for the training dataloader
            eval_batch_size (int): Batch size for the test and validation data loader's
        """
        super().__init__()
        # The following saves all arguments to self.hparams (e.g. self.hparams.horizon)
        self.save_hyperparameters()
        # Set the currently non-initialized tensor datasets for training, validating, testing
        self._data_train = self._data_val = self._data_test = None

    def setup(self, stage: Optional[str] = None):
        """ Setup data. Set internal variables: self._data_train, self._data_val, self._data_test."""
        if self._data_train and self._data_val and self._data_test:
            return  # No need to setup again

        # A small auxiliary function to preprocess the netcdf4 data
        def drop_lat_lon_info(ds: xr.Dataset) -> xr.Dataset:
            """ Drop latitude and longitude coordinates so that xarray datasets can be
             concatenated/merged along (example, grid_box) instead of (lat, lon) dimensions. """
            dummy_lat = np.arange(ds.sizes['lat'])
            dummy_lon = np.arange(ds.sizes['lon'])
            return ds.assign_coords(lat=dummy_lat, lon=dummy_lon)

        # Read all 60x60 boxes into a single xarray dataset
        glob_pattern = os.path.join(self.hparams.data_dir, 'sst.day.mean.box*.nc')
        try:
            ds = xr.open_mfdataset(
                paths=glob_pattern,
                combine='nested', concat_dim='grid_box', preprocess=drop_lat_lon_info
            ).sst
        except OSError as e:
            logging.error(f"Could not open OISSTv2 data files from {glob_pattern}."
                      f" Check that the data directory is correct: {self.hparams.data_dir}")
            raise e

        # Split the dataset into training, validation, and testing
        data_splits = {
            'train': ds.sel(time=TRAIN_SLICE),
            'val': ds.sel(time=VAL_SLICE),
            'test': ds.sel(time=TEST_SLICE)
        }
        # Create a TensorDataset for each split (here, we perform the same preprocessing for each split)
        for split_name, split_data_subset in data_splits.items():
            # Split ds into inputs and targets (targets is horizon time steps ahead of inputs)
            inputs = split_data_subset.isel(time=slice(None, -self.hparams.horizon))
            targets = split_data_subset.isel(time=slice(self.hparams.horizon, None))

            # Dimensions of X and Y: (grid-box, time, lat, lon)

            def transform(x: xr.DataArray) -> torch.Tensor:
                """ Transform the input and target data to the desired format. """
                with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                    x = x.stack(examples=('time',
                                          'grid_box'))  # Merge the time and grid_box dimensions into a single example dimension (new dimensions: (examples, lat, lon))
                x = x.transpose('examples', 'lat',
                                'lon').values  # Reorder/Reshape dimensions and convert to numpy array
                x = np.expand_dims(x, axis=1)  # Add a dummy channel dimension (needed for CNNs, Transformers, etc.)
                # Dimensions of x: (examples, channel, lat, lon) = (example, 1, 60, 60)
                x = torch.from_numpy(x).float()  # Convert to PyTorch tensor
                return x

            # Transform the inputs and targets (in this case, the same transformation is applied to both)
            inputs = transform(inputs)
            targets = transform(targets)

            # Create the pytorch tensor dataset, which will return a tuple of (input, target) when indexed
            tensor_ds = TensorDataset(inputs, targets)
            setattr(self, f'_data_{split_name}', tensor_ds)  # Save the tensor dataset to self._data_{split_name}

    # ----- Data loaders -----
    # Basically, data loaders just wrap the corresponding TensorDataset's in a pytorch DataLoader (e.g. defines the batch size)
    # Important: You usually shuffle the training data, but not the validation and test data!
    def _shared_dataloader_kwargs(self) -> dict:
        return dict(num_workers=multiprocessing.cpu_count(),
                    pin_memory=True)  # Use multiprocessing and pin memory for faster data loading

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            **self._shared_dataloader_kwargs(),
        )

    def _shared_evaluation_dataloader_kwargs(self) -> dict:
        # Disable shuffling and potentially use a larger batch size for evaluation
        return dict(**self._shared_dataloader_kwargs(), batch_size=self.hparams.eval_batch_size, shuffle=False)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self._data_val, **self._shared_evaluation_dataloader_kwargs())

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self._data_test, **self._shared_evaluation_dataloader_kwargs())
