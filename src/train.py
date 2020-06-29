from src.options.train_options import TrainOptions
from src.models.pix2pix_model import Pix2PixModel
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
        '--primary_directory', primary_directory,
        '--which_model_netG', 'pretrained_resnet',
        '--resnet_depth', '50',
        '--which_direction', 'AtoB',
        '--input_nc', '1',
        '--lambda_A', '100',
        '--lr_policy', 'plateau',
        '--epoch_count', '100',
        '--load_epoch', '-1',
        '--lr_decay_iters', '15000',
        '--lr', '0.01']

opt = TrainOptions().parse(args)

dataset_dir = '{}/data'.format(primary_directory)
model = Pix2PixModel(opt, stage='training')
# print(model)

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
