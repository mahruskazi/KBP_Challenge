import pytorch_lightning as pl
from src.models.pix2pix_model import Pix2PixModel
from src.options.train_options import TrainOptions

primary_directory = '/Users/mkazi/Google Drive/KBP_Challenge'

args = ['--batchSize', '1',
        '--primary_directory', primary_directory,
        '--which_model_netG', 'pretrained_resnet',
        '--resnet_depth', '34',
        '--which_direction', 'AtoB',
        '--input_nc', '1',
        '--lambda_A', '100',
        '--lr_policy', 'none',
        '--epoch_count', '200',
        '--load_epoch', '-1',
        '--lr_decay_iters', '15000',
        '--lr', '0.01',
        '--no_perceptual_loss']
opt = TrainOptions().parse(args)

checkpoint_number = 0
checkpoint_file = '{}/checkpoints/epoch={}.ckpt'.format(primary_directory, checkpoint_number)
# checkpoint_file = '{}/checkpoints/test.ckpt'.format(primary_directory)
dataset_dir = '{}/data'.format(primary_directory)
model = Pix2PixModel.load_from_checkpoint('/Users/mkazi/Downloads/epoch=2_v2.ckpt', opt, stage='hold-out-tests')

# print(model.state_dict())
# print(model)
trainer = pl.Trainer(max_epochs=1)
trainer.test(model)