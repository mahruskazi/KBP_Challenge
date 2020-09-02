# Tests
Below I have listed the different things I have tried out

## Generator
I have tested different architectures for the generator, including models from the Pytorch medical zoo

- UNet3D
- VNet
- UnetResnet
- Resnet3D
- UnetWithSkipConnections (This is the best tested so far)

## Discriminator
I have tested the NLayerDisriminator with different number of layers. The multiscale disriminator needs to be tested to properly calculate the loss per disriminator.
The PixelDisriminator takes up too much space on the GPU to test properly, will need to be done on multiple GPUs if possible.
The main focus on this front is to ensure the discriminator is learning properly and not saturating too quickly.

The NLayerDisriminator by default doesn't use a sigmoid on the final layer, I tried testing this but couldn't find anything conclusive. Further testing should be done here.

## Hyperparameters

We can tune the learning rates for both the generator and disriminator, model gradient clipping limits, accumulated batches, lambda_A (weight for smoothed L1 loss function), frequency for generator and disriminator, amount of noise provided to the disriminator input. Using optuna for the hyperparameter search takes a long time, so each trial should be limited to a small number of epochs (<15).

If more GPUs become available, experiment with larger batch sizes