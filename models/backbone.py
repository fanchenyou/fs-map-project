import torch
import torch.nn as nn
import torch.nn.functional as F

import pretrainedmodels
from models.utils import conv2DBatchNormRelu, deconv2DBatchNormRelu


class resnet_encoder(nn.Module):
    def __init__(self, n_classes=21, in_channels=3):
        super(resnet_encoder, self).__init__()
        feat_chn = 256
        # self.feature_backbone = n_segnet_encoder(n_classes=n_classes, in_channels=in_channels)
        self.feature_backbone = pretrainedmodels.__dict__['resnet18'](num_classes=1000, pretrained=None)
        # print(self.feature_backbone)

        self.backbone_0 = self.feature_backbone.conv1
        pool = torch.nn.MaxPool2d(kernel_size=5, stride=4, padding=1)
        self.backbone_1 = nn.Sequential(self.feature_backbone.bn1, self.feature_backbone.relu,
                                        pool, self.feature_backbone.layer1)

        # self.backbone_1 = nn.Sequential(self.feature_backbone.bn1, self.feature_backbone.relu,
        #                                 self.feature_backbone.maxpool, self.feature_backbone.layer1)

        self.backbone_2 = self.feature_backbone.layer2
        self.backbone_3 = self.feature_backbone.layer3
        self.backbone_4 = self.feature_backbone.layer4
        del self.feature_backbone.last_linear

    def forward(self, inputs):
        # torch.Size([12, 3, 512, 512])
        # torch.Size([12, 64, 256, 256])
        # torch.Size([12, 64, 128, 128])
        # torch.Size([12, 128, 64, 64])
        # torch.Size([12, 256, 32, 32])
        # torch.Size([12, 512, 16, 16])

        # torch.Size([30, 3, 512, 512])
        # torch.Size([30, 64, 256, 256])
        # torch.Size([30, 64, 64, 64])
        # torch.Size([30, 128, 32, 32])
        # torch.Size([30, 256, 16, 16])
        # torch.Size([30, 512, 8, 8])

        # print(inputs.size())

        outputs = self.backbone_0(inputs)
        # print(outputs.size())

        outputs = self.backbone_1(outputs)
        # print(outputs.size())

        outputs = self.backbone_2(outputs)
        # print(outputs.size())

        outputs = self.backbone_3(outputs)
        # print(outputs.size())

        outputs = self.backbone_4(outputs)
        # print(outputs.size())
        # print()

        return outputs


class resnet_encoder_small(nn.Module):
    def __init__(self, n_classes=21, in_channels=3):
        super(resnet_encoder_small, self).__init__()
        feat_chn = 256
        # self.feature_backbone = n_segnet_encoder(n_classes=n_classes, in_channels=in_channels)
        self.feature_backbone = pretrainedmodels.__dict__['resnet18'](num_classes=1000, pretrained=None)
        # print(self.feature_backbone)

        self.backbone_0 = self.feature_backbone.conv1
        pool = torch.nn.MaxPool2d(kernel_size=5, stride=4, padding=1)
        pool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.backbone_1 = nn.Sequential(self.feature_backbone.bn1, self.feature_backbone.relu,
                                        pool, self.feature_backbone.layer1, pool2)

        # self.backbone_1 = nn.Sequential(self.feature_backbone.bn1, self.feature_backbone.relu,
        #                                 self.feature_backbone.maxpool, self.feature_backbone.layer1)

        self.backbone_2 = self.feature_backbone.layer2
        self.backbone_3 = self.feature_backbone.layer3
        self.backbone_4 = self.feature_backbone.layer4
        del self.feature_backbone.last_linear

    def forward(self, inputs):
        # torch.Size([12, 3, 512, 512])
        # torch.Size([12, 64, 256, 256])
        # torch.Size([12, 64, 128, 128])
        # torch.Size([12, 128, 64, 64])
        # torch.Size([12, 256, 32, 32])
        # torch.Size([12, 512, 16, 16])

        # torch.Size([30, 3, 512, 512])
        # torch.Size([30, 64, 256, 256])
        # torch.Size([30, 64, 64, 64])
        # torch.Size([30, 128, 32, 32])
        # torch.Size([30, 256, 16, 16])
        # torch.Size([30, 512, 8, 8])

        # print(inputs.size())

        outputs = self.backbone_0(inputs)
        #print(outputs.size())

        outputs = self.backbone_1(outputs)
        # print(outputs.size())

        outputs = self.backbone_2(outputs)
        # print(outputs.size())

        outputs = self.backbone_3(outputs)
        # print(outputs.size())

        outputs = self.backbone_4(outputs)
        #print(outputs.size())
        # print()

        return outputs


class simple_classifier(nn.Module):
    def __init__(self, n_classes=5, in_channels=512):
        super(simple_classifier, self).__init__()
        self.in_channels = in_channels

        feat_chn = 256

        self.pred = nn.Sequential(
            nn.Conv2d(self.in_channels, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d([1, 1])
            # nn.Conv2d(feat_chn, n_classes, kernel_size=3, padding=1)
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, inputs):
        # torch.Size([12, 512, 16, 16])
        # torch.Size([12, 11, 16, 16])
        # torch.Size([12, 11, 512, 512])

        # print(inputs.size())
        out = self.pred(inputs)  # [50, 128, 1, 1]
        out = out.view(out.size(0), out.size(1))
        # print(out.size(),'====')
        pred = self.fc(out)
        # pred = nn.functional.interpolate(pred, size=torch.Size([inputs.size()[2] * 64, inputs.size()[3] * 64]),
        #                                  mode='bilinear', align_corners=False)

        return pred


class simple_decoder(nn.Module):
    def __init__(self, n_classes=21, in_channels=128):
        super(simple_decoder, self).__init__()
        self.in_channels = in_channels

        feat_chn = 128

        self.pred = nn.Sequential(
            nn.Conv2d(self.in_channels, feat_chn, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_chn, n_classes, kernel_size=3, padding=1)
        )

    def forward(self, inputs):
        pred = self.pred(inputs)
        #print(pred.size()) # [3,4,16,16]
        pred = F.interpolate(pred, size=[512, 512], mode='bilinear', align_corners=False)
        return pred

    def forward_func(self, inputs, vars):
        o1 = F.conv2d(inputs, vars[0], vars[1], padding=1)
        o2 = F.relu(o1)
        o3 = F.conv2d(o2, vars[2], vars[3], padding=1)
        pred = F.interpolate(o3, size=[512, 512], mode='bilinear', align_corners=False)
        return pred


class simple_decoder_classifier(nn.Module):
    def __init__(self, n_classes=21, in_channels=128):
        super(simple_decoder_classifier, self).__init__()
        self.in_channels = in_channels

        feat_chn = 128

        self.pred = nn.Sequential(
            nn.Conv2d(self.in_channels, feat_chn, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_chn, n_classes, kernel_size=3, padding=1)
        )

    def forward(self, inputs):
        pred = self.pred(inputs)
        pred = F.adaptive_max_pool2d(pred, output_size=[1, 1])
        pred = pred.view(pred.size(0), pred.size(1))
        return pred

    def forward_func(self, inputs, vars):
        o1 = F.conv2d(inputs, vars[0], vars[1], padding=1)
        o2 = F.relu(o1)
        o3 = F.conv2d(o2, vars[2], vars[3], padding=1)
        pred = F.adaptive_max_pool2d(o3, output_size=[1, 1])
        pred = pred.view(pred.size(0), pred.size(1))
        return pred

class n_segnet_decoder(nn.Module):
    def __init__(self, in_channels=512):
        # def __init__(self, n_classes=21, in_channels=512,agent_num=5):
        super(n_segnet_decoder, self).__init__()
        self.in_channels = in_channels
        # Decoder
        self.deconv1 = deconv2DBatchNormRelu(self.in_channels, 512, k_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = conv2DBatchNormRelu(512, 256, k_size=3, stride=1, padding=1)
        self.deconv3 = conv2DBatchNormRelu(256, 128, k_size=3, stride=1, padding=1)

    def forward(self, inputs):
        outputs = self.deconv1(inputs)
        # print(outputs.size())
        outputs = self.deconv2(outputs)
        # print(outputs.size())
        outputs = self.deconv3(outputs)
        # print(outputs.size(),'---')
        return outputs
