import pytorch_lightning as pl
from src.pix2pix_model import Pix2PixModel
from options.train_options import TrainOptions

primary_directory = '/Users/mkazi/Google Drive/KBP_Challenge'

args = ['--which_model_netG', 'unet_128_3d',
        '--which_direction', 'AtoB',
        '--batchSize', '1',
        '--input_nc', '1',
        '--lambda_A', '100']
opt = TrainOptions().parse(args)

checkpoint_number = 19
checkpoint_file = '{}/checkpoints/epoch={}.ckpt'.format(primary_directory, checkpoint_number)
# checkpoint_file = '{}/checkpoints/test.ckpt'.format(primary_directory)
dataset_dir = '{}/data'.format(primary_directory)
model = Pix2PixModel.load_from_checkpoint(checkpoint_file, opt, dataset_dir, stage='hold-out-tests')

# print(model.state_dict())

trainer = pl.Trainer(max_epochs=1)
trainer.test(model)