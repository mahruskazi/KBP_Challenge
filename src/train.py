from options.train_options import TrainOptions
from src.pix2pix_model import Pix2PixModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

primary_directory = '/Users/mkazi/Google Drive/KBP_Challenge/'

checkpoints_dir = '{}/checkpoints'.format(primary_directory)
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoints_dir,
    verbose=True,
    monitor='loss',
    mode='min'
)

args = ['--batchSize', '8',
        '--which_model_netG', 'unet_128_3d',
        '--which_direction', 'AtoB',
        '--input_nc', '1',
        '--lambda_A', '100',
        '--lr_policy', 'step',
        '--epoch_count', '1',
        '--lr_decay_iters', '1']

opt = TrainOptions().parse(args)

dataset_dir = '{}/data'.format(primary_directory)
model = Pix2PixModel(opt, dataset_dir, stage='training')

# trainer = pl.Trainer(gpus=0, checkpoint_callback=checkpoint_callback, max_epochs=opt.epoch_count)
# checkpoint = '{}/best.ckpt'.format(checkpoints_dir)
trainer = pl.Trainer(gpus=0,
                     checkpoint_callback=checkpoint_callback,
                     max_epochs=opt.epoch_count,
                     check_val_every_n_epoch=10,
                     num_sanity_val_steps=0)
# trainer = pl.Trainer(resume_from_checkpoint=checkpoint,
#                      gpus=0,
#                      checkpoint_callback=checkpoint_callback,
#                      max_epochs=opt.epoch_count,
#                      check_val_every_n_epoch=0,
#                      num_sanity_val_steps=0)
trainer.fit(model)
