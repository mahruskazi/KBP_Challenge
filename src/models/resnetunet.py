import torch.nn as nn
from torch.nn import init
import torch
from src.models import resnet3d


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


class ConBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConBlock(in_channels, out_channels),
            ConBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose3d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConBlock(in_channels, out_channels)
        self.conv_block_2 = ConBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """

        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        if up_x.size()[2] != down_x.size()[2]:
            x = self.upsample(up_x)
        else:
            x = up_x
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UNetWithResnet50Encoder(nn.Module):
    DEPTH = 6

    def __init__(self, opt, n_classes=2):
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

        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(512, 512)
        # up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        # up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(256 + 512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 256, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128, out_channels=64,
                                                    up_conv_in_channels=64, up_conv_out_channels=64))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=2, out_channels=1,
                                                    up_conv_in_channels=64, up_conv_out_channels=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        init_weights(self.bridge, init_type=self.opt.init_type)
        init_weights(self.up_blocks, init_type=self.opt.init_type)
        self.final_activation = nn.Tanh()
        # self.out = nn.Conv3d(64, 1, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.final_activation(x)
        # x = self.out(x)
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x


# model = UNetWithResnet50Encoder().cuda()
# inp = torch.rand((2, 3, 512, 512)).cuda()
# out = model(inp)