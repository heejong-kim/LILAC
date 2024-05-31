import torch.nn as nn
import torch
from torchvision.models import resnet18,resnet50

class EncoderBlock3D(nn.Module):
    '''
    Modified from Encoder implementation of LNE project (https://github.com/ouyangjiahong/longitudinal-neighbourhood-embedding)
    '''
    def __init__(self, in_num_ch, out_num_ch, kernel_size=3, conv_act='leaky_relu', dropout=0, num_conv=2, pooling=nn.MaxPool3d):
        super(EncoderBlock3D, self).__init__()
        if conv_act == 'relu':
            conv_act_layer = nn.ReLU(inplace=True)
        elif conv_act == 'leaky_relu':
            conv_act_layer = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError('No implementation of ', conv_act)

        if num_conv == 1:
            self.conv = nn.Sequential(
                            nn.Conv3d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,
                            nn.Dropout3d(dropout),
                            pooling(2))

        else:
            raise ValueError('Number of conv can only be 1 or 2')

        self.init_model()

    def init_model(self):
        for layer in self.conv.children():
            if isinstance(layer, nn.Conv3d):
                for name, weight in layer.named_parameters():
                    if 'weight' in name:
                        nn.init.kaiming_normal_(weight)
                    if 'bias' in name:
                        nn.init.constant_(weight, 0.0)

    def forward(self, x):
        return self.conv(x)

class Encoder3D(nn.Module):
    def __init__(self, in_num_ch=1, num_block=4, inter_num_ch=16, kernel_size=3, conv_act='leaky_relu', num_conv=2, pooling=nn.MaxPool3d):
        super(Encoder3D, self).__init__()

        conv_blocks = []
        for i in range(num_block):
            if i == 0: # initial block
                conv_blocks.append(EncoderBlock3D(in_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv, pooling=pooling))
            elif i == (num_block-1): # last block
                conv_blocks.append(EncoderBlock3D(inter_num_ch * (2 ** (i - 1)), inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv, pooling=pooling))
            else:
                conv_blocks.append(EncoderBlock3D(inter_num_ch * (2 ** (i - 1)), inter_num_ch * (2 ** (i)), kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv, pooling=pooling))

        self.conv_blocks = nn.Sequential(*conv_blocks)


    def forward(self, x):

        for cb in self.conv_blocks:
            x = cb(x)

        return x

class EncoderBlock2D(nn.Module):
    '''
    LSSL implementation from longitudinal-neighbourhood-embedding
    '''
    def __init__(self, in_num_ch, out_num_ch, kernel_size=3, conv_act='leaky_relu', dropout=0, num_conv=2, pooling=nn.MaxPool2d):
        super(EncoderBlock2D, self).__init__()
        if conv_act == 'relu':
            conv_act_layer = nn.ReLU(inplace=True)
        elif conv_act == 'leaky_relu':
            conv_act_layer = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError('No implementation of ', conv_act)

        if num_conv == 1:
            self.conv = nn.Sequential(
                            nn.Conv2d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm2d(out_num_ch),
                            conv_act_layer,
                            nn.Dropout2d(dropout),
                            pooling(2))
        elif num_conv == 2:
            self.conv = nn.Sequential(
                            nn.Conv2d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm2d(out_num_ch),
                            conv_act_layer,
                            nn.Dropout2d(dropout),
                            nn.Conv2d(out_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm2d(out_num_ch),
                            conv_act_layer,
                            nn.Dropout2d(dropout),
                            pooling(2))
        else:
            raise ValueError('Number of conv can only be 1 or 2')

        self.init_model()

    def init_model(self):
        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                for name, weight in layer.named_parameters():
                    if 'weight' in name:
                        nn.init.kaiming_normal_(weight)
                    if 'bias' in name:
                        nn.init.constant_(weight, 0.0)

    def forward(self, x):
        return self.conv(x)

class Encoder2D(nn.Module):
    def __init__(self, in_num_ch=1, num_block=4, inter_num_ch=16, kernel_size=3, conv_act='leaky_relu', num_conv=2, dropout=False, pooling=nn.MaxPool2d):
        super(Encoder2D, self).__init__()

        dropoutlist = [0, 0.1, 0.2, 0]

        conv_blocks = []
        for i in range(num_block):
            if i < 4 and dropout:
                dropout_ratio = dropoutlist[i]
            else:
                dropout_ratio = 0

            if i == 0: # initial block
                conv_blocks.append(EncoderBlock2D(in_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=dropout_ratio, num_conv=num_conv, pooling=pooling))
            elif i == (num_block-1): # last block
                conv_blocks.append(EncoderBlock2D(inter_num_ch * (2 ** (i - 1)), inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=dropout_ratio, num_conv=num_conv, pooling=pooling))
            else:
                conv_blocks.append(EncoderBlock2D(inter_num_ch * (2 ** (i - 1)), inter_num_ch * (2 ** (i)), kernel_size=kernel_size, conv_act=conv_act, dropout=dropout_ratio, num_conv=num_conv, pooling=pooling))

        self.conv_blocks = nn.Sequential(*conv_blocks)


    def forward(self, x):

        for cb in self.conv_blocks:
            x = cb(x)

        return x

class CNNbasic2D(nn.Module):
    def __init__(self, inputsize=[64, 64], n_of_blocks=4, channels=3, initial_channel=16, num_conv = 1,
                 pooling=nn.MaxPool2d, additional_feature=0):
        super(CNNbasic2D, self).__init__()

        self.feature_image = (torch.tensor(inputsize) / (2**(n_of_blocks)))
        self.feature_channel = initial_channel
        self.encoder = Encoder2D(in_num_ch=channels, num_block=n_of_blocks, inter_num_ch=initial_channel, num_conv=num_conv, pooling=pooling)
        self.linear = nn.Linear((self.feature_channel * (self.feature_image.prod()).type(torch.int).item()) + additional_feature, 1, bias=False)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], (self.feature_channel * (self.feature_image.prod()).type(torch.int).item()))
        y = self.linear(x)
        return y

class CNNbasic3D(nn.Module):
    def __init__(self, inputsize=[128, 128, 128], channels=1, n_of_blocks=4, initial_channel=16, num_conv=1,
                 pooling=nn.AvgPool3d, additional_feature=0):
        super(CNNbasic3D, self).__init__()

        self.feature_image = (torch.tensor(inputsize) / (2**(n_of_blocks)))
        self.feature_channel = initial_channel
        self.encoder = Encoder3D(in_num_ch=channels, num_block=n_of_blocks, inter_num_ch=initial_channel, num_conv=num_conv, pooling=pooling)
        self.linear = nn.Linear((self.feature_channel * (self.feature_image.prod()).type(torch.int).item()) + additional_feature, 1, bias=False)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], (self.feature_channel * (self.feature_image.prod()).type(torch.int).item()))
        y = self.linear(x)
        return y

def get_backbone(args=None):
    assert args != None, 'arguments are required for network configurations'
    # TODO args.optional_meta type should be list
    n_of_meta = len(args.optional_meta)

    backbone_name = args.backbone_name
    if backbone_name == 'cnn_3D':
        backbone = CNNbasic3D(inputsize=args.image_size, channels=args.image_channel, additional_feature = n_of_meta)
        linear = backbone.linear
        backbone.linear = nn.Identity()
    elif backbone_name == 'cnn_2D':
        backbone = CNNbasic2D(inputsize=args.image_size, channels=args.image_channel, additional_feature = n_of_meta)
        linear = backbone.linear
        backbone.linear = nn.Identity()
    elif backbone_name == 'resnet50_2D':
        backbone = resnet50()
        if args.image_channel != 3:
            backbone.conv1 = nn.Conv2d(args.image_channel, 64, 7, 2, 3, bias=False)
        linear = nn.Linear(2048 + n_of_meta, 1, bias=False)
        backbone.fc = nn.Identity()

    elif backbone_name == 'resnet18_2D':
        backbone = resnet18()
        if args.image_channel != 3:
            backbone.conv1 = nn.Conv2d(args.image_channel, 64, 7, 2, 3, bias=False)
        linear = nn.Linear(512 + n_of_meta, 1, bias=False)
        backbone.fc = nn.Identity()
    else:
        raise NotImplementedError(f"{args.backbone_name} not implemented yet")

    return backbone, linear

class LILAC(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone, self.linear = get_backbone(args)

    def forward(self, x1, x2, meta=None):
        f = self.backbone(x1) - self.backbone(x2)
        if meta != None:
            m1, m2 = meta
            m = m1 - m2
            f = torch.concat((f, m), 1)

        return self.linear(self.backbone(f))

