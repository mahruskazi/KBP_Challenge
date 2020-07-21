import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import models
from src.models import resnet3d, resnetunet
import src.models.medicalzoo.medzoo as medzoo
import src.models.medicalzoo.losses3D as losses3D
import copy
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F

#
# Functions
#


def weights_init_normal(m):
    ''' Initializes m.weight tensors with normal dist'''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm3d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    ''' Initializes m.weight tensors with normal dist (Xavier algorithm)'''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm3d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    ''' Initializes m.weight tensors with He algorithm.'''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm3d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm3d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    # print('initialization method [{}]'.format(init_type))
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{}] is not implemented'.format(init_type))


def get_norm_layer(norm_type='instance'):
    ''' Applies batch norm or instance norm. Batch norm: normalize the output of
    every layer before applying activation function, i.e., normalize the acti-
    vations of the previous layer for each batch. In instance normalization, we
    normalize the activations of the previous layer on a per-image/data point
    basis.
    '''
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'batch_3d':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance_3d':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError(
            'normalization layer [{}] is not found'.format(norm_type))
    return norm_layer


def get_scheduler(optimizer, opt):
    ''' Rules for how to adjust the learning rate. Lambda: custom method to
    change learning rate. StepLR: learning rate decays by gamma each step size.
    Plateau: reduce once the quantity monitored has stopped decreasing.
    '''
    if opt.lr_policy == 'lambda':

        def lambda_rule(epoch):
            lr_l = 1.0 - \
                max(0, epoch + 1 + opt.epoch_count - opt.niter) / \
                float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.9, threshold=0.01, patience=opt.patience)
    elif opt.lr_policy == 'cyclic':
        scheduler = lr_scheduler.CyclicLR(optimizer,
                                          base_lr=opt.lr,
                                          max_lr=opt.lr_max,
                                          step_size_up=opt.lr_step_size,
                                          cycle_momentum=False)
    elif opt.lr_policy == 'none':
        def lambda_rule(epoch):
            return 1.0
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:
        return NotImplementedError(
            'learning rate policy [{}] is not implemented'.format(
                opt.lr_policy))
    return scheduler


def get_loss(loss_function):

    if loss_function == 'L1':
        return nn.L1Loss()
    elif loss_function == 'L2':
        return nn.MSELoss()
    elif loss_function == 'smoothed_L1':
        return nn.SmoothL1Loss()
    elif loss_function == 'wasserstein':
        return lambda x, y: x.mean() * (-2 * int(y) + 1)
    elif loss_function == 'dice':
        return losses3D.DiceLoss(sigmoid_normalization=False)
    else:
        return NotImplementedError(
            'loss function [{}] is not implemented'.format(
                loss_function))


def define_G(opt):
    ''' Parses model parameters and defines the Generator module.
    '''
    netG = None
    norm_layer = get_norm_layer(norm_type=opt.norm)
    use_dropout = not opt.no_dropout
    use_tanh = not opt.no_scaling

    if opt.which_model_netG == 'resnet_unet':
        # netG = resnetunet.UNetWithResnet50Encoder(opt)
        netG = ResNetUNet(opt)
        # init_weights(netG, init_type=opt.init_type)
    elif opt.which_model_netG == 'pretrained_resnet':
        netG = ResnetGenerator(opt)
        init_weights(netG, init_type=opt.init_type)
    elif opt.which_model_netG == 'unet_128_3d':
        netG = UnetGenerator(
            opt.input_nc,
            opt.output_nc,
            7,
            use_tanh,
            opt.ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            conv=nn.Conv3d,
            deconv=nn.ConvTranspose3d)
        init_weights(netG, init_type=opt.init_type)
    elif opt.which_model_netG == 'unet_3d':
        netG = medzoo.UNet3D(opt.input_nc, opt.output_nc)
        init_weights(netG, init_type=opt.init_type)
    elif opt.which_model_netG == 'vnet':
        netG = medzoo.VNet(in_channels=1, classes=1, elu=False)
        init_weights(netG, init_type=opt.init_type)
    elif opt.which_model_netG == 'vnet_heavy':
        netG = medzoo.VNetHeavy(in_channels=1, classes=1, elu=False)
        init_weights(netG, init_type=opt.init_type)
    elif opt.which_model_netG == 'skipdensenet':
        netG = medzoo.SkipDenseNet3D(in_channels=1, growth_rate=16, num_init_features=32, drop_rate=0.1, classes=1)
        init_weights(netG, init_type=opt.init_type)
    elif opt.which_model_netG == 'hdresnet':
        netG = medzoo.HighResNet3D(in_channels=1, classes=1)
        init_weights(netG, init_type=opt.init_type)
    else:
        raise NotImplementedError(
            'Generator model name [{}] is not recognized'.format(opt.which_model_netG))

    return netG


def define_W(input_nc,
             output_nc,
             input_nb,
             ngf,
             which_model_netW,
             norm='batch',
             use_dropout=False,
             init_type='normal'):
    ''' Parses model parameters and defines the Generator module.
    '''
    netW = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netW == 'resnet_temp':
        netW = ResnetBeamletGenerator(
            input_nc,  # input is a dose tensor nc=1
            output_nc,  # output is 1 channel beamlet vector? nc=1
            input_nb,
            ngf,
            padding_type='zero',
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            n_blocks=3)
    else:
        raise NotImplementedError(
            'Weight Generator model name [{}] is not recognized'.format(
                which_model_netW))

    init_weights(netW, init_type=init_type)
    return netW


def define_D(opt):
    ''' Parses model parameters and defines the Discriminator module.
    '''
    netD = None

    if opt.which_model_netD == 'n_layers_spectral':
        netD = NLayerDiscriminatorSpectral(opt)
    elif opt.which_model_netD == 'multiscale':
        netD = MultiscaleDiscriminator(opt)
    else:
        raise NotImplementedError(
            'Discriminator model name [{}] is not recognized'.format(
                opt.which_model_netD))

    init_weights(netD, init_type=opt.init_type)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: {}'.format(num_params))


#
# Classes
#


class ReflectionPad3d(nn.Module):
    ''' Implements 3d version of ReflectionPad2d'''
    def __init__(self, padding):
        super(ReflectionPad3d, self).__init__()
        self.padding = torch.nn.modules.utils._ntuple(6)(padding)
        self.ReflectionPad3d = torch.nn.modules.padding._ReflectionPadNd.apply

    def forward(self, input):
        x = self.ReflectionPad3d(input, self.padding)
        return x


class View(nn.Module):
    ''' Reshape data.
    '''
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)


class GANLoss(nn.Module):
    ''' Defines GAN loss func, which uses either LSGAN or regular GAN. LSGAN is
    effectively just MSELoss, but abstracts away the need to create the target
    label tensor that has the same size as the input.
    '''

    def __init__(self,
                 use_lsgan=True,
                 target_real_label=1.0,
                 target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = torch.FloatTensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                real_tensor = real_tensor.type_as(input)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                fake_tensor = fake_tensor.type_as(input)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


# Perceptual loss that uses a pretrained VGG network
class PerceptualLoss(nn.Module):
    def __init__(self, opt):
        super(PerceptualLoss, self).__init__()

        self.opt = opt
        pretrain_path = '{}/pretrained_models/resnet_{}_23dataset.pth'.format(self.opt.primary_directory, 18)
        print('loading pretrained model {}'.format(pretrain_path))
        no_cuda = not torch.cuda.is_available()

        resnet = resnet3d.resnet18(shortcut_type='A',
                                   no_cuda=no_cuda)

        pretrain = torch.load(pretrain_path, map_location=('cpu' if no_cuda else None))

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in pretrain['state_dict'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        resnet.load_state_dict(new_state_dict)
        for param in resnet.parameters():
            param.requires_grad = False

        self.model = resnet
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 128, 1.0 / 64, 1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 2, 1.0]

    def forward(self, x, y):
        x_resnet, y_resnet = self.model(x), self.model(y)
        loss = 0
        for i in range(len(x_resnet)):
            loss += self.weights[i] * self.criterion(x_resnet[i], y_resnet[i].detach())
        return loss


# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class ResnetBeamletGenerator(nn.Module):
    ''' Defines a weight generator comprised of half of a Resnet followed

    Init:
        - Reflection Pad: 3px on all sides
        - Conv2d: input -> 64 channels, 7x7 kernels, no padding
        - Normalize -> ReLU

    Downsample:
        - Conv2d: 64 -> 128 channels, 3x3 kernels, 2 stride, 1 padding
        - Normalize -> ReLU
        - Conv2d: 128 -> 256 channels, 3x3 kernels, 2 stride, 1 padding
        - Normalize -> ReLU

    Resnet: (x n_blocks)
        - Resnet: 256 -> 256 channels (Conv -> ReLU -> Conv -> ReLU + x)

    Upsample:
        - Deconv2d: 256 -> 128 channels, 3x3 kernels, 2 stride, 1 padding
        - Normalize -> ReLU
        - Deconv2d: 128 -> 64 channels, 3x3 kernels, 2 stride, 1 padding
        - Normalize -> ReLU

    Out:
        - Reflection Pad: 3px on all sides
        - Conv2d: 64 -> output channels, 7x7 kernels, no padding
        - Tanh
    '''

    def __init__(self,
                 input_nc,
                 output_nc,
                 input_nb,
                 nwf=64,
                 norm_layer=nn.BatchNorm3d,
                 use_dropout=False,
                 n_blocks=6,
                 padding_type='reflect',
                 conv=nn.Conv3d):
        assert (n_blocks >= 0)
        super(ResnetBeamletGenerator, self).__init__()

        # input and output number of channels
        self.input_nc = input_nc
        # self.output_nc = output_nc
        self.nwf = nwf
        # bias only if we're doing instance normalization
        if type(norm_layer) == functools.partial:
            use_bias = (norm_layer.func == nn.InstanceNorm2d) or (
                norm_layer.func == nn.InstanceNorm3d)
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d) or (
                norm_layer == nn.InstanceNorm3d)

        kw = 4
        padw = 1
        model = [
            conv(input_nc, nwf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        kw = 4
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_blocks):
            # Conv2d: nwf -> 8 * nwf, 4x4 kernel size on first, the next 2
            # are 8 * nwf -> 8 * nwf, 4x4 kernel, then all subsequent ones
            # are 2 ** n * nwf -> 2 ** (n+1) * nwf , 4x4 kernel
            # each Conv followed with a norm_layer and LeakyReLU
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            model += [
                conv(
                    nwf * nf_mult_prev,
                    nwf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias),
                norm_layer(nwf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # Final Conv2d: 2 ** n * ndf -> 1, 4x4 kernels
        model += [
            conv(nwf * nf_mult, input_nb, kernel_size=kw, stride=1, padding=padw),
            norm_layer(input_nb),
            nn.LeakyReLU(0.2, True)
        ]

        n_nodes_final = 256
        model += [
            View(input_nb, -1),
            nn.Linear(15 ** 3, n_nodes_final),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(n_nodes_final, input_nb),
        ]

        model += [
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class UnetGenerator(nn.Module):
    ''' Defines the UNet generator architecture proposed in Isola et al.
    Architecture contains downsampling/upsampling operations, with Unet
    connectors.

    - Outermost Unet:
        - Conv2d: InputC -> 64, 4x4 kernels

    - Outer Unets:
        - Leaky ReLU (iv)
        - Conv2d: 64 -> 128, 4x4 kernels
        - Leaky ReLU (iii)
        - Conv2d: 128 -> 256, 4x4 kernels
        - Leaky ReLU (ii)
        - Conv2d: 256 -> 512, 4x4 kernels

    - Intermediate Unets (x num_downs):
        - Leaky ReLU (i)
        - Conv2d: 512 -> 512, 4x4 kernels

    - Inner Unet:
        - Leaky ReLU (a)
        - Conv2d: 512 -> 512, 4x4 kernels
        - ReLU
        - Deconv2d: 512 -> 512, 4x4 kernels
        - Normalize --> Connect to (a)

    - Intermediate Unets continued:
        - ReLU
        - Deconv2d: 1024 -> 512, 4x4 kernels
        - Normalize (-> Dropout ->) --> Connect to (i)

    - Outer Unets:
        - ReLU
        - Deconv2d: 512 -> 256, 4x4 kernels
        - Normalize (-> Dropout ->) --> Connect to (ii)
        - ReLU
        - Deconv2d: 256 -> 128, 4x4 kernels
        - Normalize (-> Dropout ->) --> Connect to (iii)
        - ReLU
        - Deconv2d: 128 -> 64, 4x4 kernels
        - Normalize (-> Dropout ->) --> Connect to (iv)

    - Outermost Unet:
        - ReLU
        - Deconv2d: 128 -> outC, 4x4 kernels
        - Tanh
    '''

    def __init__(self,
                 input_nc,
                 output_nc,
                 num_downs,
                 use_tanh,
                 ngf=64,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout=False,
                 conv=nn.Conv2d,
                 deconv=nn.ConvTranspose2d):
        '''
        Args:
            num_downs:  number of downsamplings in the Unet.
        '''
        super(UnetGenerator, self).__init__()

        # blocks are built recursively starting from innermost moving out
        unet_block = UnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
            conv=conv,
            deconv=deconv)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
                conv=conv,
                deconv=deconv)
        unet_block = UnetSkipConnectionBlock(
            ngf * 4,
            ngf * 8,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            conv=conv,
            deconv=deconv)
        unet_block = UnetSkipConnectionBlock(
            ngf * 2,
            ngf * 4,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            conv=conv,
            deconv=deconv)
        unet_block = UnetSkipConnectionBlock(
            ngf,
            ngf * 2,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            conv=conv,
            deconv=deconv)
        unet_block = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            use_tanh=use_tanh,
            norm_layer=norm_layer,
            conv=conv,
            deconv=deconv)

        self.model = unet_block

    def forward(self, input_data):
        return self.model(input_data)


class UnetSkipConnectionBlock(nn.Module):
    ''' Unet Skip Connection built recursively by taking a submodule and
    generating a downsample and upsample block over it. These blocks are then
    connected in a short circuit.
    '''

    def __init__(self,
                 outer_nc,
                 inner_nc,
                 input_nc=None,
                 submodule=None,
                 outermost=False,
                 innermost=False,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout=False,
                 use_tanh=True,
                 conv=nn.Conv2d,
                 deconv=nn.ConvTranspose2d):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        # bias only if we're doing instance normalization
        if type(norm_layer) == functools.partial:
            use_bias = (norm_layer.func == nn.InstanceNorm2d) or (
                norm_layer.func == nn.InstanceNorm3d)
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d) or (
                norm_layer == nn.InstanceNorm3d)
        if input_nc is None:
            input_nc = outer_nc
        # basic building blocks
        # Conv2d: inputC -> innerC, 4x4 kernel size, 2 stride, 1 padding
        downconv = conv(
            input_nc,
            inner_nc,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            # Conv2d: inputC -> innerC, 4x4 kernel size, 2 stride, 1 padding
            # then submodule
            # ReLU -> Deconv2d: innerC*2 -> outerC, 4x4 kernel size, ...
            # Tanh
            upconv = deconv(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            if use_tanh:
                up = [uprelu, upconv, nn.Tanh()]
            else:
                up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            # LeakyReLU -> Conv2d
            # ReLU -> Deconv2d: innerC -> outerC, 4x4 kernel size, 2 stride...
            # Normalize
            upconv = deconv(
                inner_nc,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            # LeakyReLU -> Conv2d -> Normalize
            # then submodule
            # ReLU -> Deconv2d -> Normalize
            upconv = deconv(
                inner_nc * 2,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            # Apply connections if inner modules
            # TODO: might need to check if cat dimension needs to change
            return torch.cat([x, self.model(x)], 1)


class ResNetUNet(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        pretrain_path = '{}/pretrained_models/resnet_{}_23dataset.pth'.format(self.opt.primary_directory, self.opt.resnet_depth)
        print('loading pretrained model {}'.format(pretrain_path))
        no_cuda = not torch.cuda.is_available()

        if opt.resnet_depth == 18:
            resnet = resnet3d.resnet18(sample_input_W=128,
                                       sample_input_H=128,
                                       sample_input_D=128,
                                       shortcut_type='A',
                                       no_cuda=no_cuda,
                                       num_seg_classes=2)
        elif opt.resnet_depth == 34:
            resnet = resnet3d.resnet34(sample_input_W=128,
                                       sample_input_H=128,
                                       sample_input_D=128,
                                       shortcut_type='A',
                                       no_cuda=no_cuda,
                                       num_seg_classes=2)
        elif opt.resnet_depth == 10:
            resnet = resnet3d.resnet10(sample_input_W=128,
                                       sample_input_H=128,
                                       sample_input_D=128,
                                       shortcut_type='B',
                                       no_cuda=no_cuda,
                                       num_seg_classes=2)

        pretrain = torch.load(pretrain_path, map_location=('cpu' if no_cuda else None))

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in pretrain['state_dict'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        resnet.load_state_dict(new_state_dict)
        for param in resnet.parameters():
            param.requires_grad = False

        self.base_layers = list(resnet.children())

        self.resnet_layer0 = nn.Sequential(*self.base_layers[:3])
        self.resnet_layer1 = nn.Sequential(*self.base_layers[3:5])
        self.resnet_layer2 = self.base_layers[5]
        self.resnet_layer3 = self.base_layers[6]
        # self.resnet_layer4 = self.base_layers[7]

        # self.unet_layer0_1x1 = self._convrelu(64, 64, 1, 0)
        # self.unet_layer1_1x1 = self._convrelu(64, 64, 1, 0)
        # self.unet_layer2_1x1 = self._convrelu(128, 128, 1, 0)
        # self.unet_layer3_1x1 = self._convrelu(256, 256, 1, 0)
        # self.unet_layer4_1x1 = self._convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        # self.upsample1 = self._convtranspose(256, 256, 4, 1)
        # self.upsample2 = self._convtranspose(256, 256, 4, 1)
        # self.upsample3 = self._convtranspose(128, 128, 4, 1)

        # self.conv_up3 = self._convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = self._convrelu(128 + 256, 256, 3, 1)
        self.conv_up1 = self._convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = self._convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = self._convrelu(1, 64, 3, 1)
        self.conv_original_size1 = self._convrelu(64, 64, 3, 1)
        self.conv_original_size2 = self._convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Sequential(
            nn.Conv3d(64, 1, 1),
            nn.BatchNorm3d(1)
        )
        init_weights(self.conv_last, init_type=self.opt.init_type)
        self.final_activation = nn.Tanh()

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        # x_original = self.conv_original_size1(x_original)

        layer0 = self.resnet_layer0(input)
        layer1 = self.resnet_layer1(layer0)
        layer2 = self.resnet_layer2(layer1)
        layer3 = self.resnet_layer3(layer2)
        # layer4 = self.resnet_layer4(layer3)

        """
        layer0 => torch.Size([1, 64, 64, 64, 64])
        layer1 => torch.Size([1, 64, 32, 32, 32])
        layer2 => torch.Size([1, 128, 16, 16, 16])
        layer3 => torch.Size([1, 256, 16, 16, 16])
        layer4 => torch.Size([1, 512, 16, 16, 16])
        """

        # layer4 = self.unet_layer4_1x1(layer4)
        # layer3 = self.unet_layer3_1x1(layer3)
        # x = torch.cat([layer4, layer3], dim=1)  # torch.Size([1, 512, 16, 16, 16])
        # x = self.conv_up3(x)

        # layer2 = self.unet_layer2_1x1(layer2)
        x = torch.cat([layer2, layer3], dim=1)
        x = self.conv_up2(x)  # torch.Size([1, 256, 16, 16, 16])

        x = self.upsample(x)  # torch.Size([1, 256, 32, 32, 32])
        # layer1 = self.unet_layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        # layer0 = self.unet_layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        output = self.conv_last(x)
        output = self.final_activation(output)

        return output

    def _convrelu(self, in_channels, out_channels, kernel, padding):
        net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel, padding=padding),
            nn.ReLU(inplace=True)
        )
        init_weights(net, init_type=self.opt.init_type)
        return net

    def _convnorm(self, in_channels, out_channels, kernel, padding):
        net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        init_weights(net, init_type=self.opt.init_type)
        return net

    def _convtranspose(self, in_channels, out_channels, kernel, padding):
        net = nn.Sequential(
            nn.ConvTranspose3d(in_channels,
                               out_channels,
                               kernel_size=kernel,
                               stride=2,
                               padding=padding,
                               bias=False),
            nn.BatchNorm3d(out_channels)
        )
        init_weights(net, init_type=self.opt.init_type)
        return net


class ResnetGenerator(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        pretrain_path = '{}/pretrained_models/resnet_{}_23dataset.pth'.format(self.opt.primary_directory, self.opt.resnet_depth)
        print('loading pretrained model {}'.format(pretrain_path))
        no_cuda = not torch.cuda.is_available()

        if opt.resnet_depth == 18:
            self.resnet = resnet3d.resnet18(sample_input_W=128,
                                            sample_input_H=128,
                                            sample_input_D=128,
                                            shortcut_type='A',
                                            no_cuda=no_cuda,
                                            num_seg_classes=2)
        elif opt.resnet_depth == 34:
            self.resnet = resnet3d.resnet34(sample_input_W=128,
                                            sample_input_H=128,
                                            sample_input_D=128,
                                            shortcut_type='A',
                                            no_cuda=no_cuda,
                                            num_seg_classes=2)

        pretrain = torch.load(pretrain_path, map_location='cpu')

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in pretrain['state_dict'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        self.resnet.load_state_dict(new_state_dict)
        for param in self.resnet.parameters():
            param.requires_grad = False

        end_layers = []
        end_layers += [self._convnorm(512, 256, 4, 1)]
        end_layers += [self._convnorm(256, 128, 4, 1)]
        end_layers += [self._convnorm(128, 64, 4, 1)]
        end_layers += [nn.Conv3d(64, 1, 1)]

        self.up_sample = nn.Sequential(*end_layers)
        # init_weights(self.up_sample, init_type=self.opt.init_type)
        self.final_activation = nn.Tanh()

    def forward(self, input):
        output = self.resnet(input)
        output = self.up_sample(output)
        output = self.final_activation(output)

        return output

    def _convnorm(self, in_channels, out_channels, kernel, padding):
        net = nn.Sequential(
            nn.ConvTranspose3d(in_channels,
                               out_channels,
                               kernel_size=kernel,
                               stride=2,
                               padding=padding,
                               bias=False),
            nn.BatchNorm3d(out_channels)
        )
        return net


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt):
        return NLayerDiscriminatorSpectral(opt)

    def downsample(self, input):
        return F.avg_pool3d(input, kernel_size=3,
                            stride=2, padding=1,
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        # get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            # if not get_intermediate_features:
            #     out = [out]
            result.append(out)
            input = self.downsample(input)

        return result


class NLayerDiscriminatorSpectral(nn.Module):
    ''' PatchGAN discriminator, supposed to work on patches within the full image
    to evaluate whether the style is transferred everywhere.
    '''

    def __init__(self, opt):
        super(NLayerDiscriminatorSpectral, self).__init__()
        self.opt = opt

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = self.opt.ndf
        input_nc = 2

        norm_layer = self.get_nonspade_norm_layer(opt, 'spectralbatch')
        sequence = [[nn.Conv3d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv3d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def get_nonspade_norm_layer(self, opt, norm_type='instance'):
        # helper function to get # output channels of the previous layer
        def get_out_channel(layer):
            if hasattr(layer, 'out_channels'):
                return getattr(layer, 'out_channels')
            return layer.weight.size(0)

        # this function will be returned
        def add_norm_layer(layer):
            nonlocal norm_type
            if norm_type.startswith('spectral'):
                layer = spectral_norm(layer)
                subnorm_type = norm_type[len('spectral'):]

            if subnorm_type == 'none' or len(subnorm_type) == 0:
                return layer

            # remove bias in the previous layer, which is meaningless
            # since it has no effect after normalization
            if getattr(layer, 'bias', None) is not None:
                delattr(layer, 'bias')
                layer.register_parameter('bias', None)

            if subnorm_type == 'batch':
                norm_layer = nn.BatchNorm3d(get_out_channel(layer), affine=True)
            elif subnorm_type == 'instance':
                norm_layer = nn.InstanceNorm3d(get_out_channel(layer), affine=False)
            else:
                raise ValueError('normalization layer %s is not recognized' % subnorm_type)

            return nn.Sequential(layer, norm_layer)

        return add_norm_layer

    def compute_D_input_nc(self, opt):
        input_nc = opt.label_nc + opt.output_nc
        if opt.contain_dontcare_label:
            input_nc += 1
        if not opt.no_instance:
            input_nc += 1
        return input_nc

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        # get_intermediate_features = not self.opt.no_ganFeat_loss
        # if get_intermediate_features:
        #     return results[1:]
        # else:
        #     return results[-1]
        return results[1:]


class PixelDiscriminator(nn.Module):
    ''' Pixel-based rather than patches.

    Architecture:
        - Conv2d: inputC -> ndf, 1x1 kernels
        - Normalize -> Leaky ReLU
        - Conv2d: ndf -> ndf * 2, 1x1 kernels
        - Normalize -> Leaky ReLU
        - Conv2d: ndf * 2 -> 1, 1x1 kernels
        - Sigmoid
    '''

    def __init__(self,
                 input_nc,
                 ndf=64,
                 norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False,
                 conv=nn.Conv2d):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = (norm_layer.func == nn.InstanceNorm2d) or (
                norm_layer.func == nn.InstanceNorm3d)
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d) or (
                norm_layer == nn.InstanceNorm3d)

        self.net = [
            conv(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            conv(
                ndf,
                ndf * 2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            conv(ndf * 2, 1, kernel_size=1, strid=1, padding=0, bias=use_bias)
        ]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input_data):
        return self.model(input_data)


class UnetCNNGenerator(nn.Module):
    ''' Defines the UNet-CNN from that one paper.
    Architecture contains downsampling/upsampling operations, with Unet
    connectors.

    - Outermost Unet:
        - Conv2d: InputC -> 64, 4x4 kernels

    - Outer Unets:
        - Leaky ReLU (iv)
        - Conv2d: 64 -> 128, 4x4 kernels
        - Leaky ReLU (iii)
        - Conv2d: 128 -> 256, 4x4 kernels
        - Leaky ReLU (ii)
        - Conv2d: 256 -> 512, 4x4 kernels

    - Intermediate Unets (x num_downs):
        - Leaky ReLU (i)
        - Conv2d: 512 -> 512, 4x4 kernels

    - Inner Unet:
        - Leaky ReLU (a)
        - Conv2d: 512 -> 512, 4x4 kernels
        - ReLU
        - Deconv2d: 512 -> 512, 4x4 kernels
        - Normalize --> Connect to (a)

    - Intermediate Unets continued:
        - ReLU
        - Deconv2d: 1024 -> 512, 4x4 kernels
        - Normalize (-> Dropout ->) --> Connect to (i)

    - Outer Unets:
        - ReLU
        - Deconv2d: 512 -> 256, 4x4 kernels
        - Normalize (-> Dropout ->) --> Connect to (ii)
        - ReLU
        - Deconv2d: 256 -> 128, 4x4 kernels
        - Normalize (-> Dropout ->) --> Connect to (iii)
        - ReLU
        - Deconv2d: 128 -> 64, 4x4 kernels
        - Normalize (-> Dropout ->) --> Connect to (iv)

    - Outermost Unet:
        - ReLU
        - Deconv2d: 128 -> outC, 4x4 kernels
        - Tanh
    '''

    def __init__(self,
                 input_nc,
                 output_nc,
                 num_downs,
                 ngf=64,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout=False,
                 conv=nn.Conv2d,
                 deconv=nn.ConvTranspose2d):
        '''
        Args:
            num_downs:  number of downsamplings in the Unet.
        '''
        super(UnetCNNGenerator, self).__init__()

        # blocks are built recursively starting from innermost moving out
        unet_block = UnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
            conv=conv,
            deconv=deconv)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
                conv=conv,
                deconv=deconv)
        unet_block = UnetSkipConnectionBlock(
            ngf * 4,
            ngf * 8,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            conv=conv,
            deconv=deconv)
        unet_block = UnetSkipConnectionBlock(
            ngf * 2,
            ngf * 4,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            conv=conv,
            deconv=deconv)
        unet_block = UnetSkipConnectionBlock(
            ngf,
            ngf * 2,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            conv=conv,
            deconv=deconv)
        unet_block = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
            conv=conv,
            deconv=deconv)

        self.model = unet_block

    def forward(self, input_data):
        return self.model(input_data)

