import numpy as np
from numpy import random
import torch
import torch.nn as nn
import math
import numbers
from torch.nn import functional as F


class RandomFlip(object):
    """ Randomly flip the data horizontally,
    """

    def __init__(self, flip=True):
        self.flip = flip

    def __call__(self, sample):
        if self.flip:
            if random.randint(2) == 0:
                sample['ct'] = sample['ct'].flip(2)
                sample['dose'] = sample['dose'].flip(2)
                sample['possible_dose_mask'] = sample['possible_dose_mask'].flip(2)
            return sample
        else:
            return sample


class RandomAugment(object):
    def __init__(self, mask_size, augment=True):
        self.augment = augment
        self.cut_blur = CutBlur(mask_size)

    def __call__(self, sample):
        mode = random.randint(3)

        if self.augment:
            if mode == 0:
                sample['ct'] = sample['ct'].flip(2)
                sample['dose'] = sample['dose'].flip(2)
                sample['possible_dose_mask'] = sample['possible_dose_mask'].flip(2)
            elif mode == 1:
                sample = self.cut_blur(sample)
            return sample

        return sample


class CutBlur(object):
    def __init__(self, mask_size):
        self.mask_size = mask_size
        self.blur = GaussianSmoothing(channels=1, kernel_size=3, sigma=1.0, dim=3)

    def __call__(self, sample):
        if self.mask_size != 0:
            ct_image = sample['ct']
            blured_image = self.blur(sample)['ct']

            x_start = random.randint(128 + 1 - self.mask_size)
            y_start = random.randint(128 + 1 - self.mask_size)
            z_start = random.randint(128 + 1 - self.mask_size)

            if random.randint(2) == 0:
                ct_image[x_start:x_start+self.mask_size][y_start:y_start+self.mask_size][z_start:z_start+self.mask_size] = \
                    blured_image[x_start:x_start+self.mask_size][y_start:y_start+self.mask_size][z_start:z_start+self.mask_size]
                sample['ct'] = ct_image
            else:
                blured_image[x_start:x_start+self.mask_size][y_start:y_start+self.mask_size][z_start:z_start+self.mask_size] = \
                    ct_image[x_start:x_start+self.mask_size][y_start:y_start+self.mask_size][z_start:z_start+self.mask_size]
                sample['ct'] = blured_image

        return sample


class GaussianSmoothing(object):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        self.kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        # self.register_buffer('weight', kernel)
        self.groups = channels

        self.reflection_pad = nn.ReplicationPad3d(1)

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def __call__(self, sample):
        ct_image = self.reflection_pad(sample['ct'].view(1, 1, 128, 128, 128))
        ct_image = self.conv(ct_image, weight=self.kernel, groups=self.groups)
        sample['ct'] = ct_image.view(128, 128, 128)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        tensored_data = {}
        for key in sample:
            valid_keys = ['ct', 'dose', 'possible_dose_mask']
            if key in valid_keys:
                tensored_data[key] = torch.from_numpy(sample[key]).view(128, 128, 128).float()
            else:
                tensored_data[key] = sample[key]

        return tensored_data


class ToRightShape(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        tensored_data = {}
        for key in sample:
            valid_keys = ['ct', 'dose', 'possible_dose_mask']
            if key in valid_keys:
                tensored_data[key] = sample[key].view(1, 128, 128, 128)
            else:
                tensored_data[key] = sample[key]
        # print(tensored_data['ct'][0][60][60])
        return tensored_data
