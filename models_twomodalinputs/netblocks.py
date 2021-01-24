import torch.nn as nn
import torch

def conv3x3(input_channel, output_channel):
    return nn.Sequential(nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1),
                         nn.BatchNorm2d(output_channel),
                         nn.ReLU(inplace=True))

def UNet_up_conv_bn_relu(input_channel, output_channel, learned_bilinear=False):

    if learned_bilinear:
        return nn.Sequential(nn.ConvTranspose2d(input_channel, output_channel, kernel_size=2, stride=2),
                             nn.BatchNorm2d(output_channel),
                             nn.ReLU())
    else:
        return nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                             nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1),
                             nn.BatchNorm2d(output_channel),
                             nn.ReLU())

class basic_block(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(basic_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class Channel_Attention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Channel_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

class Feature_refine_block(nn.Module):
    def __init__(self, in_channel):
        super(Feature_refine_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv2 = nn.Conv2d(in_channel, in_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = y + x
        y = self.relu(y)
        return y

class Spatial_Attention(nn.Module):
    def __init__(self, input_channel, reduction=16, dilation=4):
        super(Spatial_Attention, self).__init__()

        self.conv1 = nn.Conv2d(input_channel, input_channel // reduction, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(input_channel // reduction, input_channel // reduction, kernel_size=3, dilation=dilation,
                               stride=1, padding=dilation)
        self.conv3 = nn.Conv2d(input_channel // reduction, input_channel // reduction, kernel_size=3, dilation=dilation,
                               stride=1, padding=dilation)
        self.conv4 = nn.Conv2d(input_channel // reduction, 1, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.bn(self.conv4(y))
        y = self.sigmoid(y)

        return y

class Bottleneck_Attention_Module(nn.Module):
    def __init__(self, input_channel, reduction=16, dilation=4):
        super(Bottleneck_Attention_Module, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(input_channel, input_channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(input_channel // reduction, input_channel),
                nn.Sigmoid()
        )

        self.conv1 = nn.Conv2d(input_channel, input_channel // reduction, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(input_channel // reduction, input_channel // reduction, kernel_size=3, dilation=dilation, stride=1, padding=dilation)
        self.conv3 = nn.Conv2d(input_channel // reduction, input_channel // reduction, kernel_size=3, dilation=dilation, stride=1, padding=dilation)
        self.conv4 = nn.Conv2d(input_channel // reduction, 1, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.avg_pool(x).view(b, c)
        y1 = self.fc(y1).view(b, c, 1, 1)
        ca_weights = torch.ones(x.size()).cuda() * y1

        y2 = self.conv1(x)
        y2 = self.conv2(y2)
        y2 = self.conv3(y2)
        y2 = self.bn(self.conv4(y2))

        sa_weights = y2.repeat(1, x.size()[1], 1, 1)

        y = self.sigmoid(ca_weights + sa_weights)

        return y

class UNet_basic_down_block(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(UNet_basic_down_block, self).__init__()
        self.block = basic_block(input_channel, output_channel)

    def forward(self, x):
        x = self.block(x)
        return x

class UNet_basic_up_block(nn.Module):
    def __init__(self, input_channel, prev_channel, output_channel, learned_bilinear=False):
        super(UNet_basic_up_block, self).__init__()
        self.bilinear_up = UNet_up_conv_bn_relu(input_channel, prev_channel, learned_bilinear)
        self.block = basic_block(prev_channel*2, output_channel)

    def forward(self, pre_feature_map, x):
        x = self.bilinear_up(x)
        x = torch.cat((x, pre_feature_map), dim=1)
        x = self.block(x)
        return x

class UNet_ca_up_block(nn.Module):
    def __init__(self, input_channel, prev_channel, output_channel, learned_bilinear=False):
        super(UNet_ca_up_block, self).__init__()
        self.bilinear_up = UNet_up_conv_bn_relu(input_channel, prev_channel, learned_bilinear)
        self.block = basic_block(prev_channel*2, output_channel)
        self.ca = Channel_Attention(prev_channel * 2, reduction=16)

    def forward(self, pre_feature_map, x):
        x = self.bilinear_up(x)
        x = torch.cat((x, pre_feature_map), dim=1)
        x = self.ca(x) * x
        x = self.block(x)
        return x

class UNet_resca_up_block(nn.Module):
    def __init__(self, input_channel, prev_channel, output_channel, learned_bilinear=False):
        super(UNet_resca_up_block, self).__init__()
        self.bilinear_up = UNet_up_conv_bn_relu(input_channel, prev_channel, learned_bilinear)
        self.block = basic_block(prev_channel*2, output_channel)
        self.ca = Channel_Attention(prev_channel * 2, reduction=16)

    def forward(self, pre_feature_map, x):
        x = self.bilinear_up(x)
        x = torch.cat((x, pre_feature_map), dim=1)
        x = self.ca(x) * x + x
        x = self.block(x)
        return x