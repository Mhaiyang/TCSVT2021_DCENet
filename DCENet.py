"""
 @Time    : 2021/7/8 10:04
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : TCSVT2021_DCENet
 @File    : DCENet.py
 @Function: DCENet
 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import backbone.resnet.resnet as resnet

###################################################################
# ######################### SeE Block #############################
###################################################################
class SeE_Block(nn.Module):
    def __init__(self, channel):
        super(SeE_Block, self).__init__()
        self.channel = channel
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Conv2d(self.channel, self.channel, 1, 1, 0)
        self.fc2 = nn.Conv2d(self.channel, self.channel, 1, 1, 0)

    def forward(self, x):
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        fc1 = self.fc1(avg_pool)
        fc1 = self.relu(fc1)
        fc2 = self.fc2(fc1)
        fc2 = self.sigmoid(fc2)

        see = x * fc2

        return see

###################################################################
# ######################### DCE Module ############################
###################################################################
class DCE_Module(nn.Module):
    def __init__(self, input_channels):
        super(DCE_Module, self).__init__()
        self.input_channels = input_channels
        self.concat_channels = int(input_channels * 2)
        self.channels_single = int(input_channels / 4)
        self.channels_double = int(input_channels / 2)

        self.local_conv = nn.Sequential(
            nn.Conv2d(self.input_channels, self.input_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.input_channels), nn.ReLU())

        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p1_d1 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, (3, 1), 1, padding=(1, 0)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(),
            nn.Conv2d(self.channels_single, self.channels_single, (1, 3), 1, padding=(0, 1)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p1_d2 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, (1, 3), 1, padding=(0, 1)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(),
            nn.Conv2d(self.channels_single, self.channels_single, (3, 1), 1, padding=(1, 0)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p1_fusion = nn.Sequential(nn.Conv2d(self.channels_double, self.channels_single, 1, 1, 0),
                                       nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p1_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p2_d1 = nn.Sequential(
            nn.Conv2d(self.channels_double, self.channels_single, (5, 1), 1, padding=(2, 0)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(),
            nn.Conv2d(self.channels_single, self.channels_single, (1, 5), 1, padding=(0, 2)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_d2 = nn.Sequential(
            nn.Conv2d(self.channels_double, self.channels_single, (1, 5), 1, padding=(0, 2)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(),
            nn.Conv2d(self.channels_single, self.channels_single, (5, 1), 1, padding=(2, 0)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_fusion = nn.Sequential(nn.Conv2d(self.channels_double, self.channels_single, 1, 1, 0),
                                       nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p3_d1 = nn.Sequential(
            nn.Conv2d(self.channels_double, self.channels_single, (7, 1), 1, padding=(3, 0)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(),
            nn.Conv2d(self.channels_single, self.channels_single, (1, 7), 1, padding=(0, 3)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_d2 = nn.Sequential(
            nn.Conv2d(self.channels_double, self.channels_single, (1, 7), 1, padding=(0, 3)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(),
            nn.Conv2d(self.channels_single, self.channels_single, (7, 1), 1, padding=(3, 0)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_fusion = nn.Sequential(nn.Conv2d(self.channels_double, self.channels_single, 1, 1, 0),
                                       nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=7, dilation=7),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p4_d1 = nn.Sequential(
            nn.Conv2d(self.channels_double, self.channels_single, (9, 1), 1, padding=(4, 0)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(),
            nn.Conv2d(self.channels_single, self.channels_single, (1, 9), 1, padding=(0, 4)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_d2 = nn.Sequential(
            nn.Conv2d(self.channels_double, self.channels_single, (1, 9), 1, padding=(0, 4)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(),
            nn.Conv2d(self.channels_single, self.channels_single, (9, 1), 1, padding=(4, 0)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_fusion = nn.Sequential(nn.Conv2d(self.channels_double, self.channels_single, 1, 1, 0),
                                       nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=9, dilation=9),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.fusion = nn.Sequential(nn.Conv2d(self.concat_channels, self.input_channels, 1, 1, 0),
                                    nn.BatchNorm2d(self.input_channels), nn.ReLU())

    def forward(self, x):
        p1_input = self.p1_channel_reduction(x)
        p1 = self.p1_fusion(torch.cat((self.p1_d1(p1_input), self.p1_d2(p1_input)), 1))
        p1 = self.p1_dc(p1)

        p2_input = torch.cat((self.p2_channel_reduction(x), p1), 1)
        p2 = self.p2_fusion(torch.cat((self.p2_d1(p2_input), self.p2_d2(p2_input)), 1))
        p2 = self.p2_dc(p2)

        p3_input = torch.cat((self.p3_channel_reduction(x), p2), 1)
        p3 = self.p3_fusion(torch.cat((self.p3_d1(p3_input), self.p3_d2(p3_input)), 1))
        p3 = self.p3_dc(p3)

        p4_input = torch.cat((self.p4_channel_reduction(x), p3), 1)
        p4 = self.p4_fusion(torch.cat((self.p4_d1(p4_input), self.p4_d2(p4_input)), 1))
        p4 = self.p4_dc(p4)

        local_conv = self.local_conv(x)

        dce = self.fusion(torch.cat((p1, p2, p3, p4, local_conv), 1))

        return dce

###################################################################
# ########################## NETWORK ##############################
###################################################################
class DCENet(nn.Module):
    def __init__(self, backbone_path=None):
        super(DCENet, self).__init__()
        # params
        self.sigmoid = nn.Sigmoid()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # backbone
        resnet50 = resnet.resnet50(backbone_path)
        self.layer0 = nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu)
        self.layer1 = nn.Sequential(resnet50.maxpool, resnet50.layer1)
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4

        # semantic enhancement
        self.see4 = SeE_Block(2048)
        self.see3 = SeE_Block(1024)
        self.see2 = SeE_Block(512)
        self.see1 = SeE_Block(256)
        self.see0 = SeE_Block(64)

        # spatial enhancement
        self.spe4 = nn.Conv2d(2048, 1, 7, 1, 3)
        self.spe3 = nn.Conv2d(1024, 1, 7, 1, 3)
        self.spe2 = nn.Conv2d(512, 1, 7, 1, 3)
        self.spe1 = nn.Conv2d(256, 1, 7, 1, 3)

        # up
        self.up43 = nn.Sequential(nn.Conv2d(2048, 1024, 3, 1, 1), nn.BatchNorm2d(1024), nn.ReLU(), self.up)
        self.up32 = nn.Sequential(nn.Conv2d(1024, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), self.up)
        self.up21 = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), self.up)
        self.up10 = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), self.up)

        # channel reduction
        self.cr43 = nn.Sequential(nn.Conv2d(2048, 1024, 1, 1, 0), nn.BatchNorm2d(1024), nn.ReLU())
        self.cr32 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1, 0), nn.BatchNorm2d(512), nn.ReLU())
        self.cr21 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0), nn.BatchNorm2d(256), nn.ReLU())
        self.cr10 = nn.Sequential(nn.Conv2d(128, 64, 1, 1, 0), nn.BatchNorm2d(64), nn.ReLU())

        # dense context exploration
        self.dce4 = DCE_Module(2048)
        self.dce3 = DCE_Module(1024)
        self.dce2 = DCE_Module(512)
        self.dce1 = DCE_Module(256)
        self.dce0 = DCE_Module(64)

        # predict
        self.predict0 = nn.Conv2d(64, 1, 3, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        # x: [batch_size, channel=3, h, w]

        layer0 = self.layer0(x)  # [-1, 64, h/2, w/2]
        layer1 = self.layer1(layer0)  # [-1, 256, h/4, w/4]
        layer2 = self.layer2(layer1)  # [-1, 512, h/8, w/8]
        layer3 = self.layer3(layer2)  # [-1, 1024, h/16, w/16]
        layer4 = self.layer4(layer3)  # [-1, 2048, h/32, w/32]

        # 4     2048
        see4 = self.see4(layer4)
        dce4 = self.dce4(see4)
        spe4_map = self.sigmoid(self.up(self.spe4(dce4)))

        # 3     1024
        see3 = self.see3(layer3)
        spe3 = spe4_map * see3
        up43 = self.up43(dce4)
        cr43 = self.cr43(torch.cat((spe3, up43), 1))
        dce3 = self.dce3(cr43)
        spe3_map = self.sigmoid(self.up(self.spe3(dce3)))

        # 2     512
        see2 = self.see2(layer2)
        spe2 = spe3_map * see2
        up32 = self.up32(dce3)
        cr32 = self.cr32(torch.cat((spe2, up32), 1))
        dce2 = self.dce2(cr32)
        spe2_map = self.sigmoid(self.up(self.spe2(dce2)))

        # 1     256
        see1 = self.see1(layer1)
        spe1 = spe2_map * see1
        up21 = self.up21(dce2)
        cr21 = self.cr21(torch.cat((spe1, up21), 1))
        dce1 = self.dce1(cr21)
        spe1_map = self.sigmoid(self.up(self.spe1(dce1)))

        # 0     64
        see0 = self.see0(layer0)
        spe0 = spe1_map * see0
        up10 = self.up10(dce1)
        cr10 = self.cr10(torch.cat((spe0, up10), 1))
        dce0 = self.dce0(cr10)
        predict0 = self.predict0(dce0)

        # rescale to original size
        predict0 = F.upsample(predict0, size=x.size()[2:], mode='bilinear', align_corners=True)

        return self.sigmoid(predict0)
