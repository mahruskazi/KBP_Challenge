import numpy as np
from numpy import random


class RandomFlip(object):
    """ Randomly flip the data horizontally,
    """

    def __init__(self, flip=True):
        self.flip = flip

    def __call__(self, sample):
        if self.flip:
            if random.randint(2) == 0:
                ct_image = np.flip(sample['ct'].reshape((128, 128, 128)), 2).copy()
                dose = np.flip(sample['dose'].reshape((128, 128, 128)), 2).copy()
                dose_mask = np.flip(sample['possible_dose_mask'].reshape((128, 128, 128)), 2).copy()

                sample['ct'] = ct_image.reshape((1, 128, 128, 128, 1))
                sample['dose'] = dose.reshape((1, 128, 128, 128, 1))
                sample['possible_dose_mask'] = dose_mask.reshape((1, 128, 128, 128, 1))

            return sample
        else:
            return sample
