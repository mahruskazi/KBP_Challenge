import pytorch_lightning as pl
from models.pix2pix_model import Pix2PixModel
from options.train_options import TrainOptions

primary_directory = '/Users/mkazi/Google Drive/KBP_Challenge'

args = ['--which_model_netG', 'unet_128_3d',
        '--which_direction', 'AtoB',
        '--batchSize', '1',
        '--input_nc', '1',
        '--lambda_A', '100']
opt = TrainOptions().parse(args)

checkpoint_number = 0
checkpoint_file = '{}/checkpoints/epoch={}.ckpt'.format(primary_directory, checkpoint_number)
# checkpoint_file = '{}/checkpoints/test.ckpt'.format(primary_directory)
dataset_dir = '{}/data'.format(primary_directory)
model = Pix2PixModel.load_from_checkpoint('/Users/mkazi/Downloads/epoch=33.ckpt', opt, dataset_dir, stage='hold-out-tests')

# print(model.state_dict())
print(model)
# trainer = pl.Trainer(max_epochs=1)
# trainer.test(model)