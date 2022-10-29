# Graduate Climate Conference 2022 <br> Climate + Machine Learning Workshop

## Demo notebook

In the tutorial, we will focus on a self-contained demo notebook 
named [GCC-Climate-ML-workshop.ipynb](GCC-Climate-ML-workshop.ipynb), and walk through it together.
The easiest way of running the notebook is to use Google Colab, which also provides you with GPU compute.
Please follow [this link](https://githubtocolab.com/salvaRC/GCC2022-climate-machine-learning-workshop/blob/master/GCC-Climate-ML-workshop.ipynb) to open the notebook in Colab.


## Set up your own project
While the self-contained notebook above provides a good starting point, 
to start your own project or simply tinker around a bit more with the code, you may want
to modularize the code a bit more, use more advanced features, tools, etc.
In this case, you can follow the instructions below to explore the notebook problem a bit more, or set up your own project.


### Environment
Please follow the instructions in the [environment](environment/) folder to set up the correct conda environment.


### Training a model

From the repository root, please run the following in the command line:    

    python run.py trainer.gpus=0 model=simple_cnn logger=none callbacks=default

This will train an MLP on the CPU using some default callbacks and hyperparameters, but no logging.
To change the used data directory you can override the flag ``datamodule.data_dir=YOUR-DATA-DIR``.

****For more configuration & training options, please see the [src/README](src/README.md).***
