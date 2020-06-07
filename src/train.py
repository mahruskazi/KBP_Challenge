from options.train_options import TrainOptions
from pix2pix_model import Pix2PixModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from provided_code.general_functions import get_paths
import sys
import numpy as np

# opt = TrainOptions().parser.parse_args(['--print_freq=', '1'])
args = ['--which_model_netG', 'unet_128_3d',
        '--which_direction', 'AtoB',
        '--batchSize', '1',
        '--input_nc', '1',
        '--lambda_A', '100']
opt = TrainOptions().parse(args)

checkpoint_callback = ModelCheckpoint(
    filepath='/Users/mkazi/repos/python_projects/pytorch/checkpoints',
    verbose=True,
    monitor='loss',
    mode='min'
)

# model = Pix2PixModel(opt, stage='training')
model = Pix2PixModel.load_from_checkpoint('/Users/mkazi/Downloads/_ckpt_epoch_1.ckpt', opt, stage='hold-out-tests')

trainer = pl.Trainer(checkpoint_callback=checkpoint_callback, max_epochs=1)
trainer.test(model)
# trainer.fit(model)

