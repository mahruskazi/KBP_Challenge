from src.dataloaders.kbp_dataset import KBPDataset
from torch.utils.data import DataLoader
import torch
from provided_code.general_functions import get_paths
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from src.models import networks, resnet3d, resnetunet
from src.options.train_options import TrainOptions
from torchsummary import summary
import src.models.medicalzoo.medzoo as medzoo


primary_directory = '/Users/mkazi/Google Drive/KBP_Challenge'

dataset_dir = '{}/data'.format(primary_directory)
training_data_dir = '{}/train-pats'.format(dataset_dir)
# training_data_dir = '{}/validation-pats-no-dose'.format(dataset_dir)

plan_paths = get_paths(training_data_dir, ext='')  # gets the path of each plan's directory
num_train_pats = np.minimum(100, len(plan_paths))  # number of plans that will be used to train model
training_paths = plan_paths[:num_train_pats]


dataset = KBPDataset(plan_paths)

loader = DataLoader(dataset, batch_size=1, shuffle=False)

args = ['--batchSize', '8',
        '--primary_directory', primary_directory,
        '--which_model_netG', 'resnet_unet',
        '--resnet_depth', '10',
        '--which_direction', 'AtoB',
        '--input_nc', '1',
        '--lambda_A', '100',
        '--lr_policy', 'plateau',
        '--epoch_count', '100',
        '--load_epoch', '-1',
        '--lr_decay_iters', '15000',
        '--lr', '0.01']

opt = TrainOptions().parse(args)

# model = networks.ResNetUNet(opt)
model = medzoo.VNet(in_channels=1, classes=1)
print(model)

# for param in model.parameters():
#     print(param.requires_grad)
# print(model)
summary(model, (1, 128, 128, 128))

# for i, batch in enumerate(tqdm(loader)):
#     input_A = Variable(batch['ct'])
#     input_A = input_A[..., 0].float()

#     # image_3_channel = input_A.repeat(1, 3, 1, 1, 1)
#     # print(image_3_channel.size())

#     output = model(input_A)
#     print(output.size())
#     break
