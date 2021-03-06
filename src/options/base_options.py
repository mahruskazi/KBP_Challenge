import argparse
import os
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars='@')
        self.initialized = False

    def initialize(self):
        self.parser.add_argument(
                '--primary_directory',
                required=True,
                help='path to project directory')
        self.parser.add_argument(
                '--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument(
                '--loadSize',
                type=int,
                default=128,
                help='scale images to this size')
        self.parser.add_argument(
                '--fineSize', type=int, default=128, help='then crop to this size')
        self.parser.add_argument(
                '--input_nc',
                type=int,
                default=3,
                help='# of input image channels')
        self.parser.add_argument(
                '--output_nc',
                type=int,
                default=1,
                help='# of output image channels')
        self.parser.add_argument(
                '--ngf',
                type=int,
                default=64,
                help='# of gen filters in first conv layer')
        self.parser.add_argument(
                '--ndf',
                type=int,
                default=64,
                help='# of discrim filters in first conv layer')
        self.parser.add_argument(
                '--nwf',
                type=int,
                default=64,
                help='# of beamlet filters in first conv layer')
        self.parser.add_argument(
                '--which_model_netD',
                type=str,
                default='n_layers_3d',
                help='selects model to use for netD')
        self.parser.add_argument(
                '--which_model_netG',
                type=str,
                default='unet_128_3d',
                help='selects model to use for netG')
        self.parser.add_argument(
                '--which_model_netW',
                type=str,
                default='resnet_temp',
                help='selects model to use for netW')
        self.parser.add_argument(
                '--n_layers_D',
                type=int,
                default=3,
                help='only used if which_model_netD==n_layers')
        self.parser.add_argument(
                '--gpu_ids',
                type=str,
                default='-1',
                help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument(
                '--name',
                type=str,
                default='experiment_name',
                help=
                'name of the experiment. It decides where to store samples and models'
                )
        self.parser.add_argument(
                '--dataset_mode',
                type=str,
                default='unaligned',
                help=
                'chooses how datasets are loaded. [voxel | slice | aligned | unaligned | single]'
                )
        self.parser.add_argument(
                '--model',
                type=str,
                default='pix2pix',
                help='chooses which model to use. pix2pix,vox2vox, beamlet'
                )
        self.parser.add_argument(
                '--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument(
                '--nThreads',
                type=int,
                default=2,
                help='# threads for loading data')
        self.parser.add_argument(
                '--checkpoints_dir',
                type=str,
                default='./checkpoints',
                help='models are saved here')
        self.parser.add_argument(
                '--norm',
                type=str,
                default='batch_3d',
                help=
                'instance normalization or batch normalization [batch | instance | batch_3d | instance_3d]'
                )
        self.parser.add_argument(
                '--serial_batches',
                action='store_true',
                help=
                'if true, takes images in order to make batches, otherwise takes them randomly'
                )
        self.parser.add_argument(
                '--display_winsize',
                type=int,
                default=128,
                help='display window size')
        self.parser.add_argument(
                '--display_id',
                type=int,
                default=1,
                help='window id of the web display')
        self.parser.add_argument(
                '--display_port',
                type=int,
                default=8097,
                help='visdom port of the web display')
        self.parser.add_argument(
                '--no_dropout',
                action='store_true',
                help='no dropout for the generator')
        self.parser.add_argument(
                '--max_dataset_size',
                type=int,
                default=float("inf"),
                help=
                'maximum number of samples allowed per dataset. If the directory contains more than the max size, then only a subset is laoded.'
                )
        self.parser.add_argument(
                '--resize_or_crop',
                type=str,
                default='resize_and_crop',
                help=
                'scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]'
                )
        self.parser.add_argument(
                '--no_flip',
                action='store_true',
                help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument(
                '--init_type',
                type=str,
                default='normal',
                help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument(
                '--no_img',
                action='store_true',
                help=
                'if specified, do not convert 1 channel to 3 channel output.')
        self.parser.add_argument(
                '--no_normalization',
                action='store_true',
                help=
                'if specified, do not normalize the input ct images')
        self.parser.add_argument(
                '--no_scaling',
                action='store_true',
                help=
                'if specified, do not scale the reference doses to -1 to 1')

        self.initialized = True

    def parse(self, args=None):
        if not self.initialized:
            self.initialize()
        if args is None:
            self.opt = self.parser.parse_args()
        else:
            self.opt = self.parser.parse_args(args)
        self.opt.isTrain = self.isTrain  # train or test

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        return self.opt

    def mkdir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def mkdirs(self, paths):
        if isinstance(paths, list) and not isinstance(paths, str):
            for path in paths:
                self.mkdir(path)
        else:
            self.mkdir(paths)

