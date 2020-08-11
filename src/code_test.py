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
        '--which_model_netG', 'unet_128_3d',
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
dataset = KBPDataset(opt, plan_paths, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# model = networks.define_D(opt)
# model = medzoo.VNet(in_channels=1, classes=1)
# model = vae_model.VAEModel(opt)
# print(model)

# for param in model.parameters():
#     print(param.requires_grad)
# print(model)
# summary(model, (2, 128, 128, 128))

def get_loss(mask_name, structure):
    valid = torch.tensor([0], dtype=torch.float64, requires_grad=True)
    invalid = torch.tensor([1], dtype=torch.float64, requires_grad=True)

    if mask_name == 'Brainstem':
        return valid if structure.max() <= 54 else invalid
    elif mask_name == 'SpinalCord':
        return valid if structure.max() <= 48 else invalid
    elif mask_name == 'Mandible':
        return valid if structure.max() <= 73.5 else invalid
    elif mask_name == 'RightParotid':
        return valid if structure.mean() <= 26 else invalid
    elif mask_name == 'LeftParotid':
        return valid if structure.mean() <= 26 else invalid
    elif mask_name == 'Larynx':
        return valid if structure.mean() <= 45 else invalid
    elif mask_name == 'Esophagus':
        return valid if structure.mean() <= 45 else invalid
    else:
        raise Exception("%s not valid structure name" % mask_name)


for i, batch in enumerate(tqdm(loader)):
    input_A = batch['ct']

    physical_loss = {}
    print(batch['patient_list'])
    for element, mask in zip(batch['dose'], batch['structure_masks']):
        brain_mask = ((mask[..., 0]).flatten() > 0).nonzero().flatten()
        spinal_cord_mask = ((mask[..., 1]).flatten() > 0).nonzero().flatten()
        right_parotid_mask = ((mask[..., 2]).flatten() > 0).nonzero().flatten()
        left_parotid_mask = ((mask[..., 3]).flatten() > 0).nonzero().flatten()
        esophagus_mask = ((mask[..., 4]).flatten() > 0).nonzero().flatten()
        larynx_mask = ((mask[..., 5]).flatten() > 0).nonzero().flatten()
        mandible_mask = ((mask[..., 6]).flatten() > 0).nonzero().flatten()

        if brain_mask.sum() != 0:
            if 'Brainstem' in physical_loss:
                physical_loss['Brainstem'] += get_loss('Brainstem', element.flatten()[brain_mask])
            else:
                physical_loss['Brainstem'] = get_loss('Brainstem', element.flatten()[brain_mask])

        if spinal_cord_mask.sum() != 0:
            if 'SpinalCord' in physical_loss:
                physical_loss['SpinalCord'] += get_loss('SpinalCord', element.flatten()[spinal_cord_mask])
            else:
                physical_loss['SpinalCord'] = get_loss('SpinalCord', element.flatten()[spinal_cord_mask])

        if right_parotid_mask.sum() != 0:
            if 'RightParotid' in physical_loss:
                physical_loss['RightParotid'] += get_loss('RightParotid', element.flatten()[right_parotid_mask])
            else:
                physical_loss['RightParotid'] = get_loss('RightParotid', element.flatten()[right_parotid_mask])

        if left_parotid_mask.sum() != 0:
            if 'LeftParotid' in physical_loss:
                physical_loss['LeftParotid'] += get_loss('LeftParotid', element.flatten()[left_parotid_mask])
            else:
                physical_loss['LeftParotid'] = get_loss('LeftParotid', element.flatten()[left_parotid_mask])

        if esophagus_mask.sum() != 0:
            if 'Esophagus' in physical_loss:
                physical_loss['Esophagus'] += get_loss('Esophagus', element.flatten()[esophagus_mask])
            else:
                physical_loss['Esophagus'] = get_loss('Esophagus', element.flatten()[esophagus_mask])

        if larynx_mask.sum() != 0:
            if 'Larynx' in physical_loss:
                physical_loss['Larynx'] += get_loss('Larynx', element.flatten()[larynx_mask])
            else:
                physical_loss['Larynx'] = get_loss('Larynx', element.flatten()[larynx_mask])

        if mandible_mask.sum() != 0:
            if 'Mandible' in physical_loss:
                physical_loss['Mandible'] += get_loss('Mandible', element.flatten()[mandible_mask])
            else:
                physical_loss['Mandible'] = get_loss('Mandible', element.flatten()[mandible_mask])

    print(physical_loss)
    print(sum(physical_loss.values())/(len(physical_loss)))

    # ptv = batch['PTV63'][..., 0].flatten()
    # if batch['PTV63'][..., 0].flatten().sum() != 0:
    #     index = (batch['PTV63'][..., 0].flatten() > 0).nonzero()

    #     structure = batch['dose'].flatten()[index]
    #     structure = structure.flatten()
    #     size = structure.size()[0]

    #     ones = torch.ones(size)
    #     zeros = torch.zeros(size)
    #     cond = torch.where(structure > 59.9, ones, zeros)

    #     if (cond.sum()/size) < 0.95:
    #         print(structure)
    #         print("PTV56: %f" % (cond.sum()/size))
    #         print("PTV56: %f" % (cond.sum()))
    #         print(batch['patient_list'])