import os
import hydra
import torch
from omegaconf import DictConfig
from src.train import run_model


# HOW TO OVERRIDE (hyper-)parameters:
# From the command line, run e.g.:
#     python run.py model=mlp model.learning_rate=1e-4 datamodule.data_dir=/my/new/data/dir/path

@hydra.main(config_path="src/configs/", config_name="main_config.yaml", version_base=None)
def main(config: DictConfig) -> float:
    """ Run/train model based on the config file configs/main_config.yaml (and any command-line overrides). """
    return run_model(config)


if __name__ == "__main__":
    os.environ['HYDRA_FULL_ERROR'] = '1'  # show full error stack trace
    main()
