import hydra
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

import src.utilities.config_utils as cfg_utils
from src.interface import get_model_and_data


def run_model(config: DictConfig) -> float:
    r"""
    This function runs/trains/tests the model.

    .. note::
        It is recommended to call this function by running its underlying script, ``aibedo.train.py``,
        as this will enable you to make the best use of the command line integration with Hydra.
        For example, you can easily train an MLP for 10 epochs on the CPU with:

        >>>  python train.py trainer.max_epochs=10 trainer.gpus=0 model=mlp logger=none callbacks=default

    Args:
        config: A DictConfig object generated by hydra containing the model, data, callbacks & trainer configuration.

    Returns:
        float: the best model score reached while training the model.
                E.g. "val/mse", the mean squared error on the validation set.
    """
    # Seed for reproducibility
    seed_everything(config.seed)
    cfg_utils.extras(config)

    if config.get("print_config"):  # pretty print config yaml -- requires rich package to be installed
        cfg_utils.print_config(config, fields='all')

    # Obtain the instantiated model and data classes from the config
    model, datamodule = get_model_and_data(config)

    # Init Lightning callbacks and loggers (e.g. model checkpointing and Wandb logger)
    callbacks = cfg_utils.get_all_instantiable_hydra_modules(config, 'callbacks')
    loggers = cfg_utils.get_all_instantiable_hydra_modules(config, 'logger')

    # Init Lightning trainer
    trainer: pl.Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=loggers,  # , deterministic=True
    )

    # Send some parameters from config to be saved by the lightning loggers
    cfg_utils.log_hyperparameters(config=config, model=model, data_module=datamodule, trainer=trainer,
                                  callbacks=callbacks)

    trainer.fit(model=model, datamodule=datamodule)

    # Save the config to the Wandb cloud (if wandb logging is enabled)
    cfg_utils.save_hydra_config_to_wandb(config)

    # Testing:
    if config.get("test_after_training"):
        trainer.test(datamodule=datamodule, ckpt_path='best')

    if config.get('logger') and config.logger.get("wandb"):
        import wandb
        wandb.finish()

    # This is how the best model weights can be reloaded back:
    #final_model = model.load_from_checkpoint(
    #    trainer.checkpoint_callback.best_model_path,
    #    datamodule_config=config.datamodule,
    #)

    # return best score (i.e. validation mse). This is useful when using Hydra+Optuna HP tuning.
    return trainer.checkpoint_callback.best_model_score


@hydra.main(config_path="configs/", config_name="main_config.yaml", version_base=None)
def main(config: DictConfig) -> float:
    """ Run/train model based on the config file configs/main_config.yaml (and any command-line overrides). """
    return run_model(config)


if __name__ == "__main__":
    main()
