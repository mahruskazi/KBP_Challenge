from src.dataloaders.kbp_dataset import KBPDataset
from src.dataloaders.data_augmentation import RandomFlip, ToTensor, GaussianSmoothing, ToRightShape, CutBlur, RandomAugment
from torch.utils.data import DataLoader
import torch
from provided_code.general_functions import get_paths
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from src.models import networks
from src.options.train_options import TrainOptions
from torchsummary import summary
import src.models.medicalzoo.medzoo as medzoo
from torchvision import transforms


primary_directory = '/Users/mkazi/Google Drive/KBP_Challenge'

dataset_dir = '{}/data'.format(primary_directory)
training_data_dir = '{}/train-pats'.format(dataset_dir)
# training_data_dir = '{}/validation-pats-no-dose'.format(dataset_dir)

plan_paths = get_paths(training_data_dir, ext='')  # gets the path of each plan's directory
num_train_pats = np.minimum(100, len(plan_paths))  # number of plans that will be used to train model
training_paths = plan_paths[:num_train_pats]

args = ['--batchSize', '2',
        '--primary_directory', primary_directory,
        '--which_model_netG', 'pix2pixhd',
        '--which_model_netD', 'multiscale',
        '--n_layers_D', '3',
        '--num_D', '3',
        '--resnet_depth', '10',
        '--which_direction', 'AtoB',
        '--input_nc', '1',
        '--lambda_A', '100',
        '--lr_policy', 'plateau',
        '--epoch_count', '200',
        '--load_epoch', '-1',
        '--lr_G', '0.01',
        '--lr_max', '0.01',
        '--lr_step_size', '25',
        '--loss_function', 'smoothed_L1',
        '--init_type', 'xavier',
        '--no_scaling',
        '--no_normalization',
        '--patience', '5',
        '--n_critic', '5',
        '--weight_cliping_limit', '0.01']

opt = TrainOptions().parse(args)
transform = transforms.Compose([
    ToTensor(),
    # RandomAugment(opt, mask_size=64),
    # GaussianSmoothing(channels=1, kernel_size=15, sigma=3.0, dim=3),
    ToRightShape()
])

opt.batchSize = 8
print(opt.batchSize)
dataset = KBPDataset(opt, plan_paths, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

model = networks.define_G(opt)
# model = medzoo.VNet(in_channels=1, classes=1)
# model = vae_model.VAEModel(opt)
print(model)

# for param in model.parameters():
#     print(param.requires_grad)
# print(model)
# summary(model, (1, 128, 128, 128))