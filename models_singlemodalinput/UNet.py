import torch
import torch.nn as nn

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

        return x + y * x

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

################################### UNet #########################

class UNet_basic_down_block(nn.Module):
    def __init__(self, input_channel, output_channel, down_size):
        super(UNet_basic_down_block, self).__init__()
        self.block = basic_block(input_channel, output_channel)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.down_size = down_size

    def forward(self, x):
        if self.down_size:
            x = self.max_pool(x)
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

class UNet(nn.Module):
    def __init__(self, num_classes=2, learned_bilinear=False):
        super(UNet, self).__init__()

        self.down_block1 = UNet_basic_down_block(3, 64, False)
        self.down_block2 = UNet_basic_down_block(64, 128, True)
        self.down_block3 = UNet_basic_down_block(128, 256, True)
        self.down_block4 = UNet_basic_down_block(256, 512, True)
        self.down_block5 = UNet_basic_down_block(512, 1024, True)

        self.up_block1 = UNet_basic_up_block(1024, 512, 512, learned_bilinear)
        self.up_block2 = UNet_basic_up_block(512, 256, 256, learned_bilinear)
        self.up_block3 = UNet_basic_up_block(256, 128, 128, learned_bilinear)
        self.up_block4 = UNet_basic_up_block(128, 64, 64, learned_bilinear)

        self.last_conv1 = nn.Conv2d(64, num_classes, 1, padding=0)

    def forward(self, x):
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.down_block4(x3)
        x5 = self.down_block5(x4)

        x = self.up_block1(x4, x5)
        x = self.up_block2(x3, x)
        x = self.up_block3(x2, x)
        x = self.up_block4(x1, x)

        x = self.last_conv1(x)
        return x


class UNetsa(nn.Module):
    def __init__(self, num_classes=2, learned_bilinear=False):
        super(UNetsa, self).__init__()

        self.down_block1 = UNet_basic_down_block(3, 64, False)
        self.sa1 = Spatial_Attention(64, reduction=16, dilation=4)
        self.down_block2 = UNet_basic_down_block(64, 128, True)
        self.sa2 = Spatial_Attention(128, reduction=16, dilation=4)
        self.down_block3 = UNet_basic_down_block(128, 256, True)
        self.sa3 = Spatial_Attention(256, reduction=16, dilation=4)
        self.down_block4 = UNet_basic_down_block(256, 512, True)
        self.sa4 = Spatial_Attention(512, reduction=16, dilation=4)
        self.down_block5 = UNet_basic_down_block(512, 1024, True)
        self.sa5 = Spatial_Attention(1024, reduction=16, dilation=4)

        self.up_block1 = UNet_basic_up_block(1024, 512, 512, learned_bilinear)
        self.up_block2 = UNet_basic_up_block(512, 256, 256, learned_bilinear)
        self.up_block3 = UNet_basic_up_block(256, 128, 128, learned_bilinear)
        self.up_block4 = UNet_basic_up_block(128, 64, 64, learned_bilinear)

        self.last_conv1 = nn.Conv2d(64, num_classes, 1, padding=0)

    def forward(self, x):
        x1 = self.down_block1(x)
        x1 = self.sa1(x1) * x1
        x2 = self.down_block2(x1)
        x2 = self.sa2(x2) * x2
        x3 = self.down_block3(x2)
        x3 = self.sa3(x3) * x3
        x4 = self.down_block4(x3)
        x4 = self.sa4(x4) * x4
        x5 = self.down_block5(x4)
        x5 = self.sa5(x5) * x5

        x = self.up_block1(x4, x5)
        x = self.up_block2(x3, x)
        x = self.up_block3(x2, x)
        x = self.up_block4(x1, x)

        x = self.last_conv1(x)
        return x

class UNet128(nn.Module):
    def __init__(self, num_classes=2, learned_bilinear=False):
        super(UNet128, self).__init__()

        self.down_block1 = UNet_basic_down_block(3, 128, False)
        self.down_block2 = UNet_basic_down_block(128, 256, True)
        self.down_block3 = UNet_basic_down_block(256, 512, True)
        self.down_block4 = UNet_basic_down_block(512, 1024, True)
        self.down_block5 = UNet_basic_down_block(1024, 2048, True)

        self.up_block1 = UNet_basic_up_block(2048, 1024, 1024, learned_bilinear)
        self.up_block2 = UNet_basic_up_block(1024, 512, 512, learned_bilinear)
        self.up_block3 = UNet_basic_up_block(512, 256, 256, learned_bilinear)
        self.up_block4 = UNet_basic_up_block(256, 128, 128, learned_bilinear)

        self.last_conv1 = nn.Conv2d(128, num_classes, 1, padding=0)

    def forward(self, x):
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.down_block4(x3)
        x5 = self.down_block5(x4)

        x = self.up_block1(x4, x5)
        x = self.up_block2(x3, x)
        x = self.up_block3(x2, x)
        x = self.up_block4(x1, x)

        x = self.last_conv1(x)
        return x

class UNet32(nn.Module):
    def __init__(self, num_classes=2, learned_bilinear=False):
        super(UNet32, self).__init__()

        self.down_block1 = UNet_basic_down_block(3, 32, False)
        self.down_block2 = UNet_basic_down_block(32, 64, True)
        self.down_block3 = UNet_basic_down_block(64, 128, True)
        self.down_block4 = UNet_basic_down_block(128, 256, True)
        self.down_block5 = UNet_basic_down_block(256, 512, True)

        self.up_block1 = UNet_basic_up_block(512, 256, 256, learned_bilinear)
        self.up_block2 = UNet_basic_up_block(256, 128, 128, learned_bilinear)
        self.up_block3 = UNet_basic_up_block(128, 64, 64, learned_bilinear)
        self.up_block4 = UNet_basic_up_block(64, 32, 32, learned_bilinear)

        self.last_conv1 = nn.Conv2d(32, num_classes, 1, padding=0)

    def forward(self, x):
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.down_block4(x3)
        x5 = self.down_block5(x4)

        x = self.up_block1(x4, x5)
        x = self.up_block2(x3, x)
        x = self.up_block3(x2, x)
        x = self.up_block4(x1, x)

        x = self.last_conv1(x)
        return x

class UNet16(nn.Module):
    def __init__(self, num_classes=2, learned_bilinear=False):
        super(UNet16, self).__init__()

        self.down_block1 = UNet_basic_down_block(3, 16, False)
        self.down_block2 = UNet_basic_down_block(16, 32, True)
        self.down_block3 = UNet_basic_down_block(32, 64, True)
        self.down_block4 = UNet_basic_down_block(64, 128, True)
        self.down_block5 = UNet_basic_down_block(128, 256, True)

        self.up_block1 = UNet_basic_up_block(256, 128, 128, learned_bilinear)
        self.up_block2 = UNet_basic_up_block(128, 64, 64, learned_bilinear)
        self.up_block3 = UNet_basic_up_block(64, 32, 32, learned_bilinear)
        self.up_block4 = UNet_basic_up_block(32, 16, 16, learned_bilinear)

        self.last_conv1 = nn.Conv2d(16, num_classes, 1, padding=0)

    def forward(self, x):
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.down_block4(x3)
        x5 = self.down_block5(x4)

        x = self.up_block1(x4, x5)
        x = self.up_block2(x3, x)
        x = self.up_block3(x2, x)
        x = self.up_block4(x1, x)

        x = self.last_conv1(x)
        return x

class UNet8(nn.Module):
    def __init__(self, num_classes=2, learned_bilinear=False):
        super(UNet8, self).__init__()

        self.down_block1 = UNet_basic_down_block(3, 8, False)
        self.down_block2 = UNet_basic_down_block(8, 16, True)
        self.down_block3 = UNet_basic_down_block(16, 32, True)
        self.down_block4 = UNet_basic_down_block(32, 64, True)
        self.down_block5 = UNet_basic_down_block(64, 128, True)

        self.up_block1 = UNet_basic_up_block(128, 64, 64, learned_bilinear)
        self.up_block2 = UNet_basic_up_block(64, 32, 32, learned_bilinear)
        self.up_block3 = UNet_basic_up_block(32, 16, 16, learned_bilinear)
        self.up_block4 = UNet_basic_up_block(16, 8, 8, learned_bilinear)

        self.last_conv1 = nn.Conv2d(8, num_classes, 1, padding=0)

    def forward(self, x):
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.down_block4(x3)
        x5 = self.down_block5(x4)

        x = self.up_block1(x4, x5)
        x = self.up_block2(x3, x)
        x = self.up_block3(x2, x)
        x = self.up_block4(x1, x)

        x = self.last_conv1(x)
        return x

class UNet4(nn.Module):
    def __init__(self, num_classes=2, learned_bilinear=False):
        super(UNet4, self).__init__()

        self.down_block1 = UNet_basic_down_block(3, 4, False)
        self.down_block2 = UNet_basic_down_block(4, 8, True)
        self.down_block3 = UNet_basic_down_block(8, 16, True)
        self.down_block4 = UNet_basic_down_block(16, 32, True)
        self.down_block5 = UNet_basic_down_block(32, 64, True)

        self.up_block1 = UNet_basic_up_block(64, 32, 32, learned_bilinear)
        self.up_block2 = UNet_basic_up_block(32, 16, 16, learned_bilinear)
        self.up_block3 = UNet_basic_up_block(16, 8, 8, learned_bilinear)
        self.up_block4 = UNet_basic_up_block(8, 4, 4, learned_bilinear)

        self.last_conv1 = nn.Conv2d(4, num_classes, 1, padding=0)

    def forward(self, x):
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.down_block4(x3)
        x5 = self.down_block5(x4)

        x = self.up_block1(x4, x5)
        x = self.up_block2(x3, x)
        x = self.up_block3(x2, x)
        x = self.up_block4(x1, x)

        x = self.last_conv1(x)
        return x

class UNet2(nn.Module):
    def __init__(self, num_classes=2, learned_bilinear=False):
        super(UNet2, self).__init__()

        self.down_block1 = UNet_basic_down_block(3, 2, False)
        self.down_block2 = UNet_basic_down_block(2, 4, True)
        self.down_block3 = UNet_basic_down_block(4, 8, True)
        self.down_block4 = UNet_basic_down_block(8, 16, True)
        self.down_block5 = UNet_basic_down_block(16, 32, True)

        self.up_block1 = UNet_basic_up_block(32, 16, 16, learned_bilinear)
        self.up_block2 = UNet_basic_up_block(16, 8, 8, learned_bilinear)
        self.up_block3 = UNet_basic_up_block(8, 4, 4, learned_bilinear)
        self.up_block4 = UNet_basic_up_block(4, 2, 2, learned_bilinear)

        self.last_conv1 = nn.Conv2d(2, num_classes, 1, padding=0)

    def forward(self, x):
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.down_block4(x3)
        x5 = self.down_block5(x4)

        x = self.up_block1(x4, x5)
        x = self.up_block2(x3, x)
        x = self.up_block3(x2, x)
        x = self.up_block4(x1, x)

        x = self.last_conv1(x)
        return x

def test_unet():
    unet = UNet128(num_classes=2).cuda()
    print(unet)
    input = torch.ones(4, 3, 256, 256).cuda()
    x = unet(input)
    print(x.shape)
    params = list(unet.parameters())
    k = 0
    for i in params:
        I = 1
        print ('param in this layer:' + str(list(i.size())))
        for j in i.size():
            I *= j
        print('param sum in this layer:' + str(I))
        k = k + I
    print('total param:' + str(k))

# test_unet()
