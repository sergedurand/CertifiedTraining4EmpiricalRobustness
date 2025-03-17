import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def reshaper(inp, ndim):
    if inp.dim() < ndim:
        return inp.view(inp.shape + (1,) * (ndim - 1))
    else:
        return inp

def abs_forward_conv2d(cx, lay):
    # absolute value of the weights, without biases
    return F.conv2d(cx, torch.abs(lay.weight), None, lay.stride, lay.padding, lay.dilation, lay.groups)

def abs_forward_linear(cx, lay):
    # absolute value of the weights, without biases
    return F.linear(cx.reshape(cx.shape[0], -1), torch.abs(lay.weight), None)

def abs_forward_bn(inp, lirpa_bn):
    # use only the BN part affecting the multiplicative part of the linear operator
    ndim = inp.dim() - 1
    return inp / torch.sqrt(reshaper(lirpa_bn.current_var, ndim) + lirpa_bn.eps)