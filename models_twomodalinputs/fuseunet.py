import torch
import torch.nn as nn
from models_twomodalinputs.netblocks import UNet_basic_down_block, UNet_basic_up_block, Spatial_Attention
import os

class fuseunet(nn.Module):
    def __init__(self, num_classes=2, reduction=16, dilation=4, learned_bilinear=False):
        super(fuseunet, self).__init__()

        ################# modal1 encoder #################

        self.modal1_downblock1 = UNet_basic_down_block(3, 32)
        self.modal1_maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.modal1_downblock2 = UNet_basic_down_block(64, 64)
        self.modal1_maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.modal1_downblock3 = UNet_basic_down_block(128, 128)
        self.modal1_maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.modal1_downblock4 = UNet_basic_down_block(256, 256)
        self.modal1_maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.modal1_downblock5 = UNet_basic_down_block(512, 512)

        ################# modal2 encoder #################

        self.modal2_downblock1 = UNet_basic_down_block(3, 32)
        self.modal2_maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.modal2_downblock2 = UNet_basic_down_block(32, 64)
        self.modal2_maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.modal2_downblock3 = UNet_basic_down_block(64, 128)
        self.modal2_maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.modal2_downblock4 = UNet_basic_down_block(128, 256)
        self.modal2_maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.modal2_downblock5 = UNet_basic_down_block(256, 512)

        ################# decoder #################

        self.up_block1 = UNet_basic_up_block(1024, 512, 512, learned_bilinear)
        self.up_block2 = UNet_basic_up_block(512, 256, 256, learned_bilinear)
        self.up_block3 = UNet_basic_up_block(256, 128, 128, learned_bilinear)
        self.up_block4 = UNet_basic_up_block(128, 64, 64, learned_bilinear)

        self.last_conv1 = nn.Conv2d(64, num_classes, 1, padding=0)

    def forward(self, modal1_inputs, modal2_inputs):

        y = self.modal1_downblock1(modal1_inputs)

        x = self.modal2_downblock1(modal2_inputs)

        y1 = torch.cat((y, x), dim=1)

        y = self.modal1_maxpool1(y1)
        y = self.modal1_downblock2(y)

        x = self.modal2_maxpool1(x)
        x = self.modal2_downblock2(x)

        y2 = torch.cat((y, x), dim=1)

        y = self.modal1_maxpool2(y2)
        y = self.modal1_downblock3(y)

        x = self.modal2_maxpool2(x)
        x = self.modal2_downblock3(x)

        y3 = torch.cat((y, x), dim=1)

        y = self.modal1_maxpool3(y3)
        y = self.modal1_downblock4(y)

        x = self.modal2_maxpool3(x)
        x = self.modal2_downblock4(x)

        y4 = torch.cat((y, x), dim=1)

        y = self.modal1_maxpool4(y4)
        y = self.modal1_downblock5(y)

        x = self.modal2_maxpool4(x)
        x = self.modal2_downblock5(x)

        y5 = torch.cat((y, x), dim=1)

        ################# decoder #################

        y = self.up_block1(y4, y5)
        y = self.up_block2(y3, y)
        y = self.up_block3(y2, y)
        y = self.up_block4(y1, y)
        y = self.last_conv1(y)

        return y

class fuseunetsa(nn.Module):
    def __init__(self, num_classes=2, reduction=16, dilation=4, learned_bilinear=False):
        super(fuseunetsa, self).__init__()

        ################# modal1 encoder #################

        self.modal1_downblock1 = UNet_basic_down_block(3, 32)
        self.modal1_sa1 = Spatial_Attention(input_channel=32, reduction=reduction, dilation=dilation)
        self.modal1_maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.modal1_downblock2 = UNet_basic_down_block(64, 64)
        self.modal1_sa2 = Spatial_Attention(input_channel=64, reduction=reduction, dilation=dilation)
        self.modal1_maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.modal1_downblock3 = UNet_basic_down_block(128, 128)
        self.modal1_sa3 = Spatial_Attention(input_channel=128, reduction=reduction, dilation=dilation)
        self.modal1_maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.modal1_downblock4 = UNet_basic_down_block(256, 256)
        self.modal1_sa4 = Spatial_Attention(input_channel=256, reduction=reduction, dilation=dilation)
        self.modal1_maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.modal1_downblock5 = UNet_basic_down_block(512, 512)
        self.modal1_sa5 = Spatial_Attention(input_channel=512, reduction=reduction, dilation=dilation)

        ################# modal2 encoder #################

        self.modal2_downblock1 = UNet_basic_down_block(3, 32)
        self.modal2_sa1 = Spatial_Attention(input_channel=32, reduction=reduction, dilation=dilation)
        self.modal2_maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.modal2_downblock2 = UNet_basic_down_block(32, 64)
        self.modal2_sa2 = Spatial_Attention(input_channel=64, reduction=reduction, dilation=dilation)
        self.modal2_maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.modal2_downblock3 = UNet_basic_down_block(64, 128)
        self.modal2_sa3 = Spatial_Attention(input_channel=128, reduction=reduction, dilation=dilation)
        self.modal2_maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.modal2_downblock4 = UNet_basic_down_block(128, 256)
        self.modal2_sa4 = Spatial_Attention(input_channel=256, reduction=reduction, dilation=dilation)
        self.modal2_maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.modal2_downblock5 = UNet_basic_down_block(256, 512)
        self.modal2_sa5 = Spatial_Attention(input_channel=512, reduction=reduction, dilation=dilation)

        ################# decoder #################

        self.up_block1 = UNet_basic_up_block(1024, 512, 512, learned_bilinear)
        self.up_block2 = UNet_basic_up_block(512, 256, 256, learned_bilinear)
        self.up_block3 = UNet_basic_up_block(256, 128, 128, learned_bilinear)
        self.up_block4 = UNet_basic_up_block(128, 64, 64, learned_bilinear)

        self.last_conv1 = nn.Conv2d(64, num_classes, 1, padding=0)

    def forward(self, modal1_inputs, modal2_inputs):

        y = self.modal1_downblock1(modal1_inputs)
        ysa = self.modal1_sa1(y)
        y = ysa * y

        x = self.modal2_downblock1(modal2_inputs)
        xsa = self.modal2_sa1(x)
        x = xsa * x

        y1 = torch.cat((y, x), dim=1)

        y = self.modal1_maxpool1(y1)
        y = self.modal1_downblock2(y)
        ysa = self.modal1_sa2(y)
        y = ysa * y

        x = self.modal2_maxpool1(x)
        x = self.modal2_downblock2(x)
        xsa = self.modal2_sa2(x)
        x = xsa * x

        y2 = torch.cat((y, x), dim=1)

        y = self.modal1_maxpool2(y2)
        y = self.modal1_downblock3(y)
        ysa = self.modal1_sa3(y)
        y = ysa * y

        x = self.modal2_maxpool2(x)
        x = self.modal2_downblock3(x)
        xsa = self.modal2_sa3(x)
        x = xsa * x

        y3 = torch.cat((y, x), dim=1)

        y = self.modal1_maxpool3(y3)
        y = self.modal1_downblock4(y)
        ysa = self.modal1_sa4(y)
        y = ysa * y

        x = self.modal2_maxpool3(x)
        x = self.modal2_downblock4(x)
        xsa = self.modal2_sa4(x)
        x = xsa * x

        y4 = torch.cat((y, x), dim=1)

        y = self.modal1_maxpool4(y4)
        y = self.modal1_downblock5(y)
        ysa = self.modal1_sa5(y)
        y = ysa * y

        x = self.modal2_maxpool4(x)
        x = self.modal2_downblock5(x)
        xsa = self.modal2_sa5(x)
        x = xsa * x

        y5 = torch.cat((y, x), dim=1)

        ################# decoder #################

        y = self.up_block1(y4, y5)
        y = self.up_block2(y3, y)
        y = self.up_block3(y2, y)
        y = self.up_block4(y1, y)
        y = self.last_conv1(y)

        return y

class fuseunetsaseparate(nn.Module):
    def __init__(self, num_classes=2, reduction=16, dilation=4, learned_bilinear=False):
        super(fuseunetsaseparate, self).__init__()

        ################# modal1 encoder #################

        self.modal1_downblock1 = UNet_basic_down_block(3, 32)
        self.modal1_sa1 = Spatial_Attention(input_channel=32, reduction=reduction, dilation=dilation)
        self.modal1_maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.modal1_downblock2 = UNet_basic_down_block(32, 64)
        self.modal1_sa2 = Spatial_Attention(input_channel=64, reduction=reduction, dilation=dilation)
        self.modal1_maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.modal1_downblock3 = UNet_basic_down_block(64, 128)
        self.modal1_sa3 = Spatial_Attention(input_channel=128, reduction=reduction, dilation=dilation)
        self.modal1_maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.modal1_downblock4 = UNet_basic_down_block(128, 256)
        self.modal1_sa4 = Spatial_Attention(input_channel=256, reduction=reduction, dilation=dilation)
        self.modal1_maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.modal1_downblock5 = UNet_basic_down_block(256, 512)
        self.modal1_sa5 = Spatial_Attention(input_channel=512, reduction=reduction, dilation=dilation)

        ################# modal2 encoder #################

        self.modal2_downblock1 = UNet_basic_down_block(3, 32)
        self.modal2_sa1 = Spatial_Attention(input_channel=32, reduction=reduction, dilation=dilation)
        self.modal2_maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.modal2_downblock2 = UNet_basic_down_block(32, 64)
        self.modal2_sa2 = Spatial_Attention(input_channel=64, reduction=reduction, dilation=dilation)
        self.modal2_maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.modal2_downblock3 = UNet_basic_down_block(64, 128)
        self.modal2_sa3 = Spatial_Attention(input_channel=128, reduction=reduction, dilation=dilation)
        self.modal2_maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.modal2_downblock4 = UNet_basic_down_block(128, 256)
        self.modal2_sa4 = Spatial_Attention(input_channel=256, reduction=reduction, dilation=dilation)
        self.modal2_maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.modal2_downblock5 = UNet_basic_down_block(256, 512)
        self.modal2_sa5 = Spatial_Attention(input_channel=512, reduction=reduction, dilation=dilation)

        ################# decoder #################

        self.up_block1 = UNet_basic_up_block(1024, 512, 512, learned_bilinear)
        self.up_block2 = UNet_basic_up_block(512, 256, 256, learned_bilinear)
        self.up_block3 = UNet_basic_up_block(256, 128, 128, learned_bilinear)
        self.up_block4 = UNet_basic_up_block(128, 64, 64, learned_bilinear)

        self.last_conv1 = nn.Conv2d(64, num_classes, 1, padding=0)

    def forward(self, modal1_inputs, modal2_inputs):

        y = self.modal1_downblock1(modal1_inputs)
        ysa = self.modal1_sa1(y)
        y = ysa * y

        x = self.modal2_downblock1(modal2_inputs)
        xsa = self.modal2_sa1(x)
        x = xsa * x

        y1 = torch.cat((y, x), dim=1)

        y = self.modal1_maxpool1(y)
        y = self.modal1_downblock2(y)
        ysa = self.modal1_sa2(y)
        y = ysa * y

        x = self.modal2_maxpool1(x)
        x = self.modal2_downblock2(x)
        xsa = self.modal2_sa2(x)
        x = xsa * x

        y2 = torch.cat((y, x), dim=1)

        y = self.modal1_maxpool2(y)
        y = self.modal1_downblock3(y)
        ysa = self.modal1_sa3(y)
        y = ysa * y

        x = self.modal2_maxpool2(x)
        x = self.modal2_downblock3(x)
        xsa = self.modal2_sa3(x)
        x = xsa * x

        y3 = torch.cat((y, x), dim=1)

        y = self.modal1_maxpool3(y)
        y = self.modal1_downblock4(y)
        ysa = self.modal1_sa4(y)
        y = ysa * y

        x = self.modal2_maxpool3(x)
        x = self.modal2_downblock4(x)
        xsa = self.modal2_sa4(x)
        x = xsa * x

        y4 = torch.cat((y, x), dim=1)

        y = self.modal1_maxpool4(y)
        y = self.modal1_downblock5(y)
        ysa = self.modal1_sa5(y)
        y = ysa * y

        x = self.modal2_maxpool4(x)
        x = self.modal2_downblock5(x)
        xsa = self.modal2_sa5(x)
        x = xsa * x

        y5 = torch.cat((y, x), dim=1)

        ################# decoder #################

        y = self.up_block1(y4, y5)
        y = self.up_block2(y3, y)
        y = self.up_block3(y2, y)
        y = self.up_block4(y1, y)
        y = self.last_conv1(y)

        return y

if __name__ == '__main__':

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    unet = fuseunet(num_classes=2)
    print(unet)
    input = torch.ones(1, 3, 32, 32)
    x = unet(input, input)
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