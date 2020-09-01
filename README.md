# OpenKBP Challenge

![alt text](https://github.com/mahruskazi/KBP_Challenge/blob/master/images/Pytorch_logo.png?raw=true "Logo Title Text 1")

This project aims to: "implement the most accurate KBP dose prediction method on a large open-access dataset".
Using a dataset comprimised of 3D CT images that are clean and scaled to 128x128x128. The competition provided code is contained in a folder called "provided_code", which contains code to calculate the following scores for evaluation:
  - dose score
  - dvh score

### Dataset
The dataset can be found here: https://www.dropbox.com/sh/2pvtbtelsadzfcc/AADWX3YY0YwtLo9iG8BSSbqFa?dl=0
This includes data for training and validation.

### Installation
There are two ways to run this project:
1. Google Colab
2. Locally on a Linux machine with an Nvidia GPU + CUDA

If running locally, clone the repo and start a terminal session in the main project folder.
We first install the required dependencies, it is advisable to create a virtual environment first.

Note* I haven't actually run this locally due to the lack of a GPU, however this setup does work on a single GPU running CUDA version 10.1 and this version of pytorch (torch==1.6.0+cu101).
```sh
$ pip3 install -r requirements.txt
```
### Usage

##### Local
Note* Have not tested with GPU
```sh
$ python -m src.train
```

##### Google Colab
The quickest way to get started is running training on Google Colab. It is advisable to install Google Drive on your machine and clone this repo into the mounted drive. Your folder structure should look something like this:

```
Google Drive
└── KBP_Challenge
    └── src
        └── ...
    └── provided_code
    └── data
        └── train-pats
        └── validation-pats-no-dose
    └── pretrained_models
        └── ...
    └── requirements.txt
```

This allows you to change files on the fly without the need for re-uploading files manually to Google Drive.
You can also clone **train.ipynb** multiple times to run multiple experiements at the same time. Getting Google Colab Pro will significatly increase experiment runtimes and RAM limits.

Restart runtime and run all after making a change (might have to wait ~30sec to ensure the filesystem has the most up-to-date file).

### Tools
We take advatange of multiple tools to make training easier/efficient, below are some of the main ones and an explaination of how we use them.

![alt text](https://github.com/mahruskazi/KBP_Challenge/blob/master/images/PTL.png?raw=true "Pytorch Lightning")

Pytorch lightning (PTL) provides researchers a high level interface for Pytorch, allowing us to quickly start training on multiple GPUs with no change in code. The framework allows users to implement different callbacks and hooks for various points in training, definitely look at their documentation for an extensive list of options.
The Trainer has a couple of hyperparameters available for tuning:

- accumulate_grad_batches
- gradient_clip_val

PTL also comes with options for saving and loading model checkpoints utilizing the ModelCheckpoint class. Currently configured to save the model with the lowest validation dose_score after every epoch.

![alt text](https://github.com/mahruskazi/KBP_Challenge/blob/master/images/logo_comet_light.png?raw=true "Comet ML")

Comet ML provides a nice way of visualizing the different metrics created during training. The "Experiments" as Comet ML calls them live online on their website and can be accessed anytime/anywhere, making it easy to share results.
It integrates nicely with Pytorch Lightning and allows users to easily filter experiments based on differeent metrics. 

![alt text](https://github.com/mahruskazi/KBP_Challenge/blob/master/images/optuna-logo.png?raw=true "Optuna")

Optuna is an automatic hyperparameter optimization framework, it allow users to specifiy a range of values for parameters and the framework will try to find the "right" ones after a number of trials.
**hyper_param_search.ipynb** has been created to easily use the framework. The study is saved after every trial, the study contains the parameters tested and the best trial.





