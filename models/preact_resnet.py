import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import abs_forward_bn, abs_forward_linear, abs_forward_conv2d

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            downsample_conv = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            self.shortcut = nn.Sequential(downsample_conv)

        self._n_bn = 2

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

    def abs_forward(self, x, use_bn=False, lirpa_bns=None):

        if use_bn:
            # use only the BN part affecting the multiplicative part of the linear operator
            x = abs_forward_bn(x, lirpa_bns[0])

        if hasattr(self, 'shortcut'):
            shortcut = abs_forward_conv2d(x, self.shortcut._modules['0'])
        else:
            shortcut = x

        out = abs_forward_conv2d(x, self.conv1)

        if use_bn:
            # use only the BN part affecting the multiplicative part of the linear operator
            out = abs_forward_bn(out, lirpa_bns[1])

        out = abs_forward_conv2d(out, self.conv2)
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            downsample_conv = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

        self._n_bn = 3

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

    def abs_forward(self, x, use_bn=False, lirpa_bns=None):
        if use_bn:
            out = abs_forward_bn(x, lirpa_bns[0])
        else:
            out = x
        shortcut = x
        out = abs_forward_conv2d(out, self.conv1)
        if use_bn:
            out = abs_forward_bn(x, lirpa_bns[1])
        out = abs_forward_conv2d(out, self.conv2)
        if use_bn:
            out = abs_forward_bn(x, lirpa_bns[2])
        out = abs_forward_conv2d(out, self.conv3)
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.bncounter = 0

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def abs_forward(self, x, use_bn=False, lirpa_bns=None):
        # Implements a forward pass using the absolute value of the weights, yet without biases nor activation functions
        self.bncounter = 0

        out = abs_forward_conv2d(x, self.conv1)

        for c in self.layer1._modules.keys():
            out = self.layer1._modules[c].abs_forward(
                out, use_bn=use_bn,
                lirpa_bns=lirpa_bns[self.bncounter:self.bncounter + self.layer1._modules[c]._n_bn])
            self.bncounter += self.layer1._modules[c]._n_bn
        for c in self.layer2._modules.keys():
            out = self.layer2._modules[c].abs_forward(
                out, use_bn=use_bn,
                lirpa_bns=lirpa_bns[self.bncounter:self.bncounter + self.layer2._modules[c]._n_bn])
            self.bncounter += self.layer2._modules[c]._n_bn
        for c in self.layer3._modules.keys():
            out = self.layer3._modules[c].abs_forward(
                out, use_bn=use_bn,
                lirpa_bns=lirpa_bns[self.bncounter:self.bncounter + self.layer3._modules[c]._n_bn])
            self.bncounter += self.layer3._modules[c]._n_bn
        for c in self.layer4._modules.keys():
            out = self.layer4._modules[c].abs_forward(
                out, use_bn=use_bn,
                lirpa_bns=lirpa_bns[self.bncounter:self.bncounter + self.layer4._modules[c]._n_bn])
            self.bncounter += self.layer4._modules[c]._n_bn

        if use_bn:
            # use only the BN part affecting the multiplicative part of the linear operator
            out = abs_forward_bn(out, lirpa_bns[self.bncounter])

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = abs_forward_linear(out, self.linear)
        return out

def PreActResNet18(**kwargs):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], **kwargs)

