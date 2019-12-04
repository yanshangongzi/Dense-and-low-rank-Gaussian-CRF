import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def down_size(in_size):
    out_size = int(in_size)
    out_size = (out_size + 1) // 2
    out_size = int(np.ceil((out_size + 1) / 2.0))
    out_size = (out_size + 1) // 2
    return out_size


class BottleneckLayer(nn.Module):

    def build_downsample(self, use_downsample):
        if not use_downsample:
            return nn.Identity()

        if self.stride != 1 or self.in_channels != self.out_channels or self.dilation == 2 or self.dilation == 4:
            layer = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels,
                    kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(self.out_channels, affine=False)
            )
            return layer

        return nn.Identity()


    def __init__(self, in_channels, out_channels, stride=1, dilation=1, use_downsample=False):
        super(BottleneckLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        padding = dilation

        layers = [
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels // 4, affine=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=1,
                padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels // 4, affine=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=False)
        ]

        self.downsample = self.build_downsample(use_downsample)
        self.layers = nn.Sequential(*layers)
        self.activation = nn.ReLU(inplace=True)


    def forward(self, x):
        out = self.layers(x)
        out += self.downsample(x)
        out = self.activation(out)
        return out


class ClassifierLayer(nn.Module):

    def init_weights(self):
        for conv in self.convolutions:
            conv.weight.data.normal_(0, 0.01)


    def __init__(self, in_channels, dilations, paddings, n_labels):
        super(ClassifierLayer, self).__init__()
        self.convolutions = nn.ModuleList()
        for dilation, padding in zip(dilations, paddings):
            self.convolutions.append(
                nn.Conv2d(in_channels, n_labels, kernel_size=3, stride=1,
                    padding=padding, dilation=dilation, bias=True)
            )

        self.init_weights()


    def forward(self, x):
        out = self.convolutions[0](x)
        for i in range(1, len(self.convolutions)):
            out += self.convolutions[i](x)

        return out


class ResNet(nn.Module):

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0, 0.01)

    def construct_block(self, in_channels, out_channels, n_blocks, stride=1, dilation=1):
        layers = []
        layers.append(BottleneckLayer(
            in_channels, out_channels, stride, dilation, use_downsample=True
        ))

        for i in range(1, n_blocks):
            layers.append(BottleneckLayer(
                out_channels, out_channels, dilation=dilation
            ))

        return nn.Sequential(*layers)


    def __init__(self, n_blocks, n_labels):
        super(ResNet, self).__init__()

        self.first_out = 64
        self.in_layer = nn.Sequential(
            nn.Conv2d(3, self.first_out, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.first_out, affine=False),
            nn.ReLU()
        )

        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)

        self.bottlenecks = nn.Sequential(
            self.construct_block(64, 256, n_blocks[0]),
            self.construct_block(256, 512, n_blocks[1], stride=2),
            self.construct_block(512, 1024, n_blocks[2], stride=1, dilation=2),
            self.construct_block(1024, 2048, n_blocks[3], stride=1, dilation=4)
        )

        self.classifier = ClassifierLayer(2048, [6, 12, 18, 24], [6, 12, 18, 24], n_labels)

        self.init_weights()

    def forward(self, x):
        x = self.in_layer(x)
        x = self.pooling(x)
        x = self.bottlenecks(x)
        x = self.classifier(x)
        return x


class ResDeepLab(nn.Module):
    def __init__(self, n_labels=21):
        super(ResDeepLab, self).__init__()
        self.resnet = ResNet([3, 4, 23, 3], n_labels)

    def forward(self, x):
        out = []
        x_075 = F.interpolate(x, 
            size=(int(x.shape[2] * 0.75) + 1, int(x.shape[3] * 0.75) + 1),
            mode='bilinear', align_corners=True
        )

        x_05 = F.interpolate(x,
            size=(int(x.shape[2] * 0.5) + 1, int(x.shape[3] * 0.5) + 1),
            mode='bilinear', align_corners=True
        )

        out.append(self.resnet(x))
        out.append(F.interpolate(self.resnet(x_075),
            size=(down_size(x.shape[2]), down_size(x.shape[3])),
            mode='bilinear', align_corners=True)
        )
        out.append(self.resnet(x_05))

        x_05_int = F.interpolate(out[2],
            size=(down_size(x.shape[2]), down_size(x.shape[3])),
            mode='bilinear', align_corners=True
        )

        out.append(torch.max(
            torch.max(out[0], out[1]),
            x_05_int)
        )

        return out[-1]


class ResNetPW(nn.Module):

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0, 0.01)

    def construct_block(self, in_channels, out_channels, n_blocks, stride=1, dilation=1):
        layers = []
        layers.append(BottleneckLayer(
            in_channels, out_channels, stride, dilation, use_downsample=True
        ))

        for i in range(1, n_blocks):
            layers.append(BottleneckLayer(
                out_channels, out_channels, dilation=dilation
            ))

        return nn.Sequential(*layers)


    def __init__(self, n_layers, embedding_size, n_blocks, n_labels):
        super(ResNetPW, self).__init__()

        self.embedding_size = embedding_size
        self.n_labels = n_labels

        self.first_out = 64
        self.in_layer = nn.Sequential(
            nn.Conv2d(3, self.first_out, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.first_out, affine=False),
            nn.ReLU()
        )

        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)

        sizes = [256, 512, 1024, 2048]
        bottlenecks = [
            self.construct_block(64, 256, n_blocks[0]),
            self.construct_block(256, 512, n_blocks[1], stride=2),
            self.construct_block(512, 1024, n_blocks[2], stride=1, dilation=2),
            self.construct_block(1024, 2048, n_blocks[3], stride=1, dilation=4)
        ]

        self.bottlenecks = nn.Sequential(*bottlenecks[:n_layers])
        self.classifier = ClassifierLayer(sizes[n_layers - 1], [1], [1],
            n_labels * embedding_size)

        self.init_weights()


    def forward(self, x):
        x = self.in_layer(x)
        x = self.pooling(x)
        x = self.bottlenecks(x)
        x = self.classifier(x)
        x = x.reshape(x.shape[0], self.embedding_size, self.n_labels, x.shape[-2], x.shape[-1])
        return x

