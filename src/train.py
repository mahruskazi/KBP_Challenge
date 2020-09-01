import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.logging import CometLogger
from src.models.pix2pix_model import Pix2PixModel
from pytorch_lightning.callbacks import ModelCheckpoint
from src.options.train_options import TrainOptions
from torchsummary import summary

primary_directory = '/Users/mkazi/Google Drive/KBP_Challenge/'

args = ['--batchSize', '8',
        '--primary_directory', primary_directory,
        '--which_model_netG', 'unet_128_3d',
        '--which_model_netD', 'n_layers_3d',
        '--n_layers_D', '3',
        '--num_D', '1',
        '--norm_D', 'spectralbatch',
        '--norm_G', 'spectralbatch',
        '--resnet_depth', '18',
        '--which_direction', 'AtoB',
        '--input_nc', '1',
        '--lambda_A', '10',
        '--lr_policy', 'plateau',
        '--epoch_count', '300',
        '--load_epoch', '-1',
        '--lr_D', '0.0004',
        '--lr_G', '0.0001',
        '--lr_max', '0.01',
        '--lr_step_size', '25',
        '--loss_function', 'smoothed_L1',
        '--init_type', 'xavier',
        #'--no_scaling',
        # '--no_normalization',
        '--no_perceptual_loss',
        '--patience', '5',
        '--n_critic', '1',
        '--n_generator', '3',
        '--cut_blur_mask', '50',
        '--weight_cliping_limit', '0.1',
        '--inst_noise_sigma', '80',
        '--inst_noise_sigma_iters', '100']

opt = TrainOptions().parse(args)

model = Pix2PixModel(opt, model_name='train', stage='training')
summary(model.generator.to("cuda"), (1, 128, 128, 128))

checkpoints_dir = '{}/checkpoints'.format(primary_directory)
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoints_dir,
    verbose=True,
    save_last=False,
    save_top_k=0,
    monitor='dose_score',
    mode='min'
)
lr_logger = LearningRateLogger()

comet_logger = CometLogger(
    api_key="eyAsnp1KA7fXLxFMkEWKjhygS",
    project_name="kbp-challenge",
    workspace="mahruskazi"
)

checkpoint = None
if opt.load_epoch != -1:
    checkpoint = '{}/epoch={}.ckpt'.format(checkpoints_dir, opt.load_epoch)

trainer = pl.Trainer(logger=comet_logger,
                     resume_from_checkpoint=checkpoint,
                     gpus=1,
                     checkpoint_callback=checkpoint_callback,
                     callbacks=[lr_logger],
                     max_epochs=opt.epoch_count,
                     check_val_every_n_epoch=1,
                     num_sanity_val_steps=10,
                     limit_val_batches=1.0,
                     accumulate_grad_batches=1,
                     gradient_clip_val=opt.weight_cliping_limit,
                     weights_summary='full')
trainer.fit(model)
