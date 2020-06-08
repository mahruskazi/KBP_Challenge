from options.train_options import TrainOptions
from src.pix2pix_model import Pix2PixModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

primary_directory = '/Users/mkazi/Google Drive/KBP_Challenge/'
args = ['--which_model_netG', 'unet_128_3d',
        '--which_direction', 'AtoB',
        '--batchSize', '1',
        '--input_nc', '1',
        '--lambda_A', '100']
opt = TrainOptions().parse(args)

checkpoints_dir = '{}/checkpoints'.format(primary_directory)
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoints_dir,
    verbose=True,
    monitor='loss',
    mode='min'
)

dataset_dir = '{}/data'.format(primary_directory)
model = Pix2PixModel(opt, dataset_dir, stage='training')

trainer = pl.Trainer(gpus=0, checkpoint_callback=checkpoint_callback, max_epochs=2, progress_bar_refresh_rate=10)
trainer.fit(model)

