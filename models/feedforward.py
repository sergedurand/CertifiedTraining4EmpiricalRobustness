import json
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from auto_LiRPA import BoundedModule
from auto_LiRPA.operators import BoundBatchNormalization
from torch.nn.parameter import Parameter

from config import load_config
from models.preact_resnet import PreActResNet, PreActBlock
# from .utils import Flatten
from models.utils import abs_forward_bn, abs_forward_linear, abs_forward_conv2d, Flatten
import math
import pdb

def cnn_7layer(in_ch=3, in_dim=32, width=64, linear_size=512):
    model = nn.Sequential(
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear((in_dim // 2) * (in_dim // 2) * 2 * width, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 10)
    )
    return model


def cnn_7layer_bn(in_ch=3, in_dim=32, width=64, linear_size=512):
    model = nn.Sequential(
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        Flatten(),
        nn.Linear((in_dim // 2) * (in_dim // 2) * 2 * width, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 10)
    )
    return model


def cnn_7layer_bn2(in_ch=3, in_dim=32, width=64, linear_size=512, num_class=10):
    model = nn.Sequential(
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        Flatten(),
        nn.Linear((in_dim // 2) * (in_dim // 2) * 2 * width, linear_size),
        nn.BatchNorm1d(linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, num_class)
    )
    return model


def set_abs_forward(model):
    model.bncounter = 0

    def abs_forward(cx, use_bn=False, lirpa_bns=None):
        # Implements a forward pass using the absolute value of the weights, yet without biases nor activation functions
        model.bncounter = 0
        for lay in model.children():
            if isinstance(lay, nn.Linear):
                cx = abs_forward_linear(cx, lay)
            elif isinstance(lay, nn.Conv2d):
                cx = abs_forward_conv2d(cx, lay)
            elif isinstance(lay, (nn.BatchNorm2d, nn.BatchNorm1d)) and use_bn:
                cx = abs_forward_bn(cx, lirpa_bns[model.bncounter])
                model.bncounter += 1
        return cx

    model.abs_forward = abs_forward


def cnn(in_ch=3, in_dim=32, num_class=10):
    model = cnn_7layer_bn2(in_ch, in_dim, num_class=num_class)
    set_abs_forward(model)
    return model


def cnn_5layer(in_ch=3, in_dim=32, width=64, linear_size=512, num_class=10):
    # taken from TAPS, but setting width=64 instead of width=16
    model = nn.Sequential(
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, width, 4, stride=2, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 4, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim // 4) ** 2 * 2 * width, linear_size),
        nn.BatchNorm1d(linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, num_class)
    )
    return model


def cnn5(in_ch=3, in_dim=32, num_class=10):
    model = cnn_5layer(in_ch, in_dim, num_class=num_class)
    set_abs_forward(model)
    return model


def cnn_wide(in_ch=3, in_dim=32):
    return cnn_7layer_bn2(in_ch, in_dim, width=128)


def cnn_7layer_imagenet(in_ch=3, in_dim=32, width=64, linear_size=512, num_class=200):
    conv_backbone = nn.Sequential(
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
    )
    dummy = torch.rand(1, in_ch, in_dim, in_dim)
    out = conv_backbone(dummy)
    in_linear = out.shape[1]
    print('in_linear', in_linear)
    model = torch.nn.Sequential(
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(in_linear, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, num_class)
    )
    del conv_backbone
    return model


def cnn_7layer_bn_imagenet(in_ch=3, in_dim=32, width=64, linear_size=512, num_class=200):
    model = nn.Sequential(
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32768, linear_size),
        nn.BatchNorm1d(linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, num_class)
    )

    dummy = torch.rand(1, in_ch, in_dim, in_dim)
    out = model[0:16](dummy)
    in_linear = out.shape[1]
    model[16] = nn.Linear(in_linear, linear_size)
    return model


def cnn_6layer(in_ch, in_dim, width=32, linear_size=256):
    model = nn.Sequential(
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear((in_dim // 2) * (in_dim // 2) * 2 * width, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 10)
    )
    return model


# FIXME: linear_size is smaller than other models
def cnn_6layer_bn2(in_ch, in_dim, width=32, linear_size=256, num_class=10):
    model = nn.Sequential(
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        Flatten(),
        nn.Linear((in_dim // 2) * (in_dim // 2) * 2 * width, linear_size),
        nn.BatchNorm1d(linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, num_class)
    )
    return model


"""DM-large"""


def cnn_large(in_ch, in_dim, num_classes=10):
    return nn.Sequential(
        nn.Conv2d(in_ch, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(128 * (in_dim // 2) * (in_dim // 2), 512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    )
    return model


# resnet & company

def timm_resnet18(in_ch, in_dim, num_class=10):
    import timm
    model = timm.create_model('resnet18', pretrained=False, num_classes=num_class)
    dummy_input = torch.rand(4, in_ch, in_dim, in_dim)
    _ = model(dummy_input)
    return model


def preactresnet18(in_ch, in_dim, num_class=10):
    model = PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_class)
    dummy_input = torch.rand(4, in_ch, in_dim, in_dim)
    _ = model(dummy_input)
    return model


if __name__ == "__main__":
    # checking bn in forwabs.
    # fix seeds
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)
    model = preactresnet18(3, 32, 10)
    model_ori = deepcopy(model)
    dummy = torch.rand(4, 3, 32, 32)
    with open("../config/defaults.json") as f:
        config = json.load(f)

    bound_config = config['bound_params']
    bound_opts = bound_config['bound_opts']

    model = BoundedModule(model, dummy, bound_opts=bound_opts, custom_ops={}, device="cpu")

    orig_bn = [(name, param) for name, param in model_ori.named_parameters() if 'bn' in name]
    orig_bn_weights = [(name, param.data.shape) for name, param in orig_bn if 'weight' in name]
    print(len(orig_bn_weights))
    print(f"Number of original bn {len(orig_bn_weights)}")
    all_named_modules = list()
    lirpa_modules = model._modules

    lirpa_bns = list()
    lirpa_bns_ori_name = list()
    node_name_map = model.node_name_map
    for cm in lirpa_modules.values():
        if isinstance(cm, BoundBatchNormalization):
            lirpa_bns.append(cm)
            # the corresponding bn in the original model can be bound in its first BoundParams.
            # Lirpa BoundBachNorm has 5 inputs: the actual input of the batch norm,
            # its weight, bias, running mean and running var.
            # original bn weight name is inputs[1].ori_name
            ori_weight_name = cm.inputs[1].ori_name
            ori_weight_shape = cm.inputs[1].param.data.shape
            lirpa_bns_ori_name.append((ori_weight_name, ori_weight_shape))
    print(len(f"Number of lirpa bn {len(lirpa_bns)}"))
    assert orig_bn_weights == lirpa_bns_ori_name


