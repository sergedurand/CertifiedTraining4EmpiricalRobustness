import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from auto_LiRPA import BoundedModule, BoundDataParallel, BoundedTensor, CrossEntropyWrapper
from auto_LiRPA.bound_ops import *
from collections import namedtuple

from loguru import logger

from utils import get_modules
from certified import get_loss_over_lbs

Node = namedtuple('Node', 'node lower upper')


def compute_L1_reg(args, model, meter):
    loss = torch.zeros(()).to(args.device)
    for module in model._modules.values():
        if isinstance(module, nn.Linear):
            loss += torch.abs(module.weight).sum()
        elif isinstance(module, nn.Conv2d):
            loss += torch.abs(module.weight).sum()
    meter.update('L1_loss', loss)
    return loss * args.l1_coeff


def compute_reg(args, model, meter, eps, eps_scheduler):
    try:
        from lightning_fabric.wrappers import _FabricModule
        using_fabric = isinstance(model, _FabricModule)
    except ImportError:
        using_fabric = False
    loss = torch.zeros(())
    if not using_fabric:
        loss = loss.to(args.device)
    else:
        loss = loss.to(next(model.parameters()).device)
    # Handle the non-feedforward case
    l0 = torch.zeros_like(loss)
    loss_tightness, loss_std, loss_relu, loss_ratio = (l0.clone() for i in range(4))

    modules = get_modules(model)

    # if isinstance(model, BoundDataParallel):
    #     modules = list(model._modules.values())[0]._modules
    # elif isinstance(model, _FabricModule):
    #     modules = model._modules["_forward_module"]._modules
    # else:
    #     modules = model._modules
    # if "module" in modules:
    #     modules = modules["module"]._modules
    try:
        node_inp = modules['/input.1']
    except KeyError:
        print(type(modules))
        if hasattr(modules, 'keys'):
            print(modules.keys())
        print(type(model))
    # logger.info("Model device: {}", model.device)
    # logger.info("Modules device: {}", modules['/input.1'].device)
    # logger.info("Type of modules: {}", type(modules))
    # logger.info("Keys of modules: {}", modules.keys())
    tightness_0 = ((node_inp.upper - node_inp.lower) / 2).mean()
    ratio_init = tightness_0 / ((node_inp.upper + node_inp.lower) / 2).std()
    cnt_layers = 0
    cnt = 0
    for m in modules.values():
        if isinstance(m, BoundRelu):
            lower, upper = m.inputs[0].lower, m.inputs[0].upper
            center = (upper + lower) / 2
            diff = ((upper - lower) / 2)
            tightness = diff.mean()
            mean_ = center.mean()
            std_ = center.std()
            # moving everything to the proper device
            # device = m.device
            # lower, upper, center, diff, tightness, mean_, std_ = (
            #     lower.to(device), upper.to(device), center.to(device), diff.to(device),
            #     tightness.to(device), mean_.to(device), std_.to(device)
            # )
            #

            loss_tightness += F.relu(args.tol - tightness_0 / tightness.clamp(min=1e-12)) / args.tol
            loss_std += F.relu(args.tol - std_) / args.tol
            cnt += 1

            # L_{relu}
            mask_act, mask_inact = lower > 0, upper < 0
            mean_act = (center * mask_act).mean()
            mean_inact = (center * mask_inact).mean()
            delta = (center - mean_) ** 2
            var_act = (delta * mask_act).sum()  # / center.numel()
            var_inact = (delta * mask_inact).sum()  # / center.numel()

            mean_ratio = mean_act / -mean_inact
            var_ratio = var_act / var_inact
            mean_ratio = torch.min(mean_ratio, 1 / mean_ratio.clamp(min=1e-12))
            var_ratio = torch.min(var_ratio, 1 / var_ratio.clamp(min=1e-12))
            loss_relu_ = ((
                                  F.relu(args.tol - mean_ratio) + F.relu(args.tol - var_ratio))
                          / args.tol)
            if not torch.isnan(loss_relu_) and not torch.isinf(loss_relu_):
                loss_relu += loss_relu_

            if args.debug:
                bn_mean = (lower.mean() + upper.mean()) / 2
                bn_var = ((upper ** 2 + lower ** 2) / 2).mean() - bn_mean ** 2
                print(m.name, m,
                      'tightness {:.4f} gain {:.4f} std {:.4f}'.format(
                          tightness.item(), (tightness / tightness_0).item(), std_.item()),
                      'input', m.inputs[0], m.inputs[0].name,
                      'active {:.4f} inactive {:.4f}'.format(
                          (lower > 0).float().sum() / lower.numel(),
                          (upper < 0).float().sum() / lower.numel()),
                      'bnv2_mean {:.5f} bnv2_var {:.5f}'.format(bn_mean.item(), bn_var.item())
                      )
                # pre-bn
                lower, upper = m.inputs[0].inputs[0].lower, m.inputs[0].inputs[0].upper
                bn_mean = (lower.mean() + upper.mean()) / 2
                bn_var = ((upper ** 2 + lower ** 2) / 2).mean() - bn_mean ** 2
                print('pre-bn',
                      'bnv2_mean {:.5f} bnv2_var {:.5f}'.format(bn_mean.item(), bn_var.item()))

    loss_tightness /= cnt
    loss_std /= cnt
    loss_relu /= cnt

    if args.debug:
        pdb.set_trace()

    for item in ['tightness', 'relu', 'std']:
        loss_ = eval('loss_{}'.format(item))
        if item in args.reg_obj:
            loss += loss_
        meter.update('L_{}'.format(item), loss_)

    meter.update('loss_reg', loss)

    if args.no_reg_dec:
        intensity = args.reg_lambda
    else:
        intensity = args.reg_lambda * (1 - eps_scheduler.get_eps() / eps_scheduler.get_max_eps())
    loss *= intensity

    return loss


def compute_forwabs_reg(data_lb, data_ub, model, coeff, use_bn=False, lirpa_bns=None):
    # NOTE: still cheaper than ELLE
    # Regularize through an FORWard pass in ABSolute-value of the input bound difference
    return model.abs_forward((data_ub - data_lb) / 2, use_bn=use_bn, lirpa_bns=lirpa_bns).sum() * coeff


def compute_elle_reg(args, data, label, data_lb, data_ub, model):
    #  NOTE: adapted from https://github.com/LIONS-EPFL/ELLE/blob/main/core/utils/train.py
    bs = data.shape[0]
    x_ab = data.repeat([2, 1, 1, 1])

    # NOTE: changed the original code as this repository handles normalization differently
    # NOTE: differently from the original codebase, we clip to the data domain [0, 1]
    x_ab_lb = data_lb.repeat([2, 1, 1, 1])
    x_ab_ub = data_ub.repeat([2, 1, 1, 1])
    x_ab = (torch.rand(x_ab.shape, device=data.device) * (x_ab_ub - x_ab_lb) + x_ab_lb)
    # x_ab = x_ab + args.attack_eps * (2 * torch.rand(x_ab.shape, device=data.device) - 1)

    alpha = torch.rand([bs, 1, 1, 1], device=data.device)
    x_c = (1 - alpha) * x_ab[:bs] + alpha * x_ab[bs:]
    alpha = alpha.squeeze()

    # Forward pass
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    losses = criterion(model(torch.cat((x_ab, x_c), dim=0)), label.repeat([3]))

    # Regularization term
    mse = torch.nn.MSELoss()
    lin_err = mse(losses[2 * bs:], (1 - alpha) * losses[:bs] + alpha * losses[bs:2 * bs])

    return lin_err
