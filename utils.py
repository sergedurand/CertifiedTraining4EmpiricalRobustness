import random
import os
import pdb
import copy
from pathlib import Path
from typing import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from auto_LiRPA.bound_ops import BoundExp, BoundRelu, BoundBatchNormalization
from auto_LiRPA.operators import BoundDropout
from loguru import logger, logger as logging
# from auto_LiRPA.eps_scheduler import *
from schedulers.schedulers import *

from models import *
from auto_LiRPA import PerturbationLpNorm, BoundedTensor, BoundDataParallel

ce_loss = nn.CrossEntropyLoss()



def eps_handling(args, eps, std, robust, reg):
    norm_eps = eps
    if not robust and reg:
        norm_eps = max(norm_eps, args.min_eps_reg)
    if type(norm_eps) == float:
        norm_eps = (norm_eps / std).view(1, -1, 1, 1)
    else:  # [batch_size, channels]
        norm_eps = (norm_eps.view(*norm_eps.shape, 1, 1) / std.view(1, -1, 1, 1))
    return norm_eps


def compute_perturbation(args, eps, data, data_min, data_max, std, robust, reg):
    norm_eps = eps_handling(args, eps, std, robust, reg)
    if norm_eps.device != data.device:
        norm_eps = norm_eps.to(data.device)
    if data_min.device != data.device:
        data_min = data_min.to(data.device)
    if data_max.device != data.device:
        data_max = data_max.to(data.device)

    data_ub = torch.min(data + norm_eps, data_max)
    data_lb = torch.max(data - norm_eps, data_min)
    ptb = PerturbationLpNorm(norm=np.inf, eps=norm_eps, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb)
    return x, data_lb, data_ub


def compute_sabr_perturbation(args, eps, data, adv_data, data_min, data_max, std, robust, reg, coef_scheduler=None):
    norm_eps = eps_handling(args, eps, std, robust, reg)
    if norm_eps.device != data.device:
        norm_eps = norm_eps.to(data.device)
    if data_min.device != data.device:
        data_min = data_min.to(data.device)
    if data_max.device != data.device:
        data_max = data_max.to(data.device)

    # norm_eps, data_min, data_max, std = norm_eps.to(data.device), data_min.to(data.device), data_max.to(
    #     data.device), std.to(data.device)
    if coef_scheduler is not None:
        coef = coef_scheduler.get_eps()
    else:
        coef = args.sabr_coeff
    norm_sabr_eps = eps_handling(args, eps * coef, std, robust, reg)
    if norm_sabr_eps.device != data.device:
        norm_sabr_eps = norm_sabr_eps.to(data.device)

    # SABR re-centering
    if "nfgsm" not in args.attack:
        data_ub = torch.min(data + norm_eps, data_max)
        data_lb = torch.max(data - norm_eps, data_min)
        sabr_center = torch.clamp(adv_data, data_lb + norm_sabr_eps, data_ub - norm_sabr_eps)
    else:
        sabr_center = adv_data
    # compute small box
    sabr_data_ub = torch.min(sabr_center + norm_sabr_eps, data_max)
    sabr_data_lb = torch.max(sabr_center - norm_sabr_eps, data_min)
    ptb = PerturbationLpNorm(norm=np.inf, eps=norm_sabr_eps, x_L=sabr_data_lb, x_U=sabr_data_ub)
    # the center of the ball is unused for IBP on l-inf perts: data is passed for consistency with the other methods
    x = BoundedTensor(data, ptb)
    return x, sabr_center


def set_file_handler(logger, dir):
    file_handler = logging.FileHandler(os.path.join(dir, 'train.log'))
    file_handler.setFormatter(logging.Formatter('%(levelname)-8s %(asctime)-12s %(message)s'))
    logger.addHandler(file_handler)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


# In loss fusion, update the state_dict of `model` from the loss fusion version
# `model_loss`. This is necessary when BatchNorm is involved.
def update_state_dict(model, model_loss):
    state_dict_loss = model_loss.state_dict()
    state_dict = model.state_dict()
    keys = model.state_dict().keys()
    for name in state_dict_loss:
        v = state_dict_loss[name]
        for prefix in ['model.', '/w.', '/b.', '/running_mean.']:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        if not name in keys:
            raise KeyError(name)
        state_dict[name] = v
    model.load_state_dict(state_dict)


def update_meter(meter, regular_ce, robust_loss, adv_loss, regular_err, robust_err, adv_err, batch_size):
    meter.update('CE', regular_ce, batch_size)
    if robust_loss is not None:
        meter.update('Rob_Loss', robust_loss, batch_size)
    if regular_err is not None:
        meter.update('Err', regular_err, batch_size)
    if robust_err is not None:
        meter.update('Rob_Err', robust_err, batch_size)
    if robust_loss is not None:
        meter.update('Rob_Loss', robust_loss, batch_size)
    if adv_err is not None:
        meter.update('Adv_Err', adv_err, batch_size)
    if adv_loss is not None:
        meter.update('Adv_Loss', adv_loss, batch_size)


def parse_opts(s):
    opts = s.split(',')
    params = {}
    for o in opts:
        if o.strip():
            key, val = o.split('=')
            try:
                v = eval(val)
            except:
                v = val
            if type(v) not in [int, float, bool]:
                v = val
            params[key] = v
    return params


def get_modules(model):
    try:
        # attempt importing _FabricModule
        from lightning_fabric.wrappers import _FabricModule
        if isinstance(model, _FabricModule):
            modules = model._modules["_forward_module"]._modules
            if isinstance(modules, OrderedDict):
                if "modules" in modules:
                    modules = modules["modules"]
            return modules
    except ImportError:
        pass

    if isinstance(model, BoundDataParallel):
        modules = list(model._modules.values())[0]._modules
    else:
        modules = model._modules
    if isinstance(modules, OrderedDict):
        if "modules" in modules:
            modules = modules["modules"]
    return modules


def prepare_model(args, logger, config):
    model = args.model
    logger.info(f'Using model {model}')
    logger.info(f'Dataset : {config["data"]}')
    if config['data'] == 'MNIST':
        input_shape = (1, 28, 28)
    elif config['data'] == 'CIFAR' or config['data'] == 'CIFAR10':
        input_shape = (3, 32, 32)
    elif config['data'] in ['tinyimagenet', 'imagenet64']:
        input_shape = (3, 64, 64)
    elif config['data'] == "CIFAR100":
        input_shape = (3, 32, 32)
    elif config['data'] == "SVHN":
        input_shape = (3, 32, 32)
    else:
        raise NotImplementedError(config['data'])

    model_ori = eval(model)(in_ch=input_shape[0], in_dim=input_shape[1],
                            num_class=args.num_class,
                            **parse_opts(args.model_params))

    checkpoint = None
    if args.auto_load:
        path_last = os.path.join(args.dir, 'ckpt_last')
        if os.path.exists(path_last):
            args.load = path_last
            logger.info('Use last checkpoint {}'.format(path_last))
        else:
            latest = -1
            for filename in os.listdir(args.dir):
                if filename.startswith('ckpt_'):
                    latest = max(latest, int(filename[5:]))
            if latest != -1:
                args.load = os.path.join(args.dir, 'ckpt_{}'.format(latest))
                try:
                    checkpoint = torch.load(args.load, map_location="cpu")
                except:
                    logger.warning('Cannot load {}'.format(args.load))
                    args.load = os.path.join(args.dir, 'ckpt_{}'.format(latest - 1))
                    logger.warning('Trying {}'.format(args.load))
    if checkpoint is None and args.load:
        checkpoint = torch.load(args.load, map_location="cpu")
    if checkpoint is not None:
        epoch, state_dict = checkpoint['epoch'], checkpoint['state_dict']
        best = checkpoint.get('best', (100., 100., -1))
        model_ori.load_state_dict(state_dict, strict=False)
        logger.info(f'Checkpoint loaded: {args.load}, epoch {epoch}')
    else:
        epoch = 0
        best = (100., 100., -1)

    return model_ori, checkpoint, epoch, best


def save(args, name_prefix, epoch, model, opt, xp_id=None):
    coef = 0.0
    if args.sabr:
        coef = args.sabr_coeff
    elif args.mtlibp:
        coef = args.mtlibp_coeff
    elif args.ccibp:
        coef = args.ccibp_coeff
    l1_coeff = args.l1_coeff
    reg_lambda = args.reg_lambda
    init_method = args.init_method

    ckpt = {
        'state_dict': model.state_dict(), 'optimizer': opt.state_dict(),
        'epoch': epoch,
        "seed": args.seed,
        "eps": args.eps,
        "coef": coef,
        "params": args,
        "l1_coeff": l1_coeff,
        "reg_lambda": reg_lambda,
        "init_method": init_method,
        "xp_id": xp_id
    }
    path_last = os.path.join(args.dir, name_prefix + 'ckpt_last')
    if os.path.exists(path_last):
        os.system('mv {path} {path}.bak'.format(path=path_last))
    torch.save(ckpt, path_last)
    if args.save_all:
        torch.save(ckpt, os.path.join(args.dir, name_prefix + 'ckpt_{}'.format(epoch)))
    logger.info('')


def get_eps_scheduler(args, max_eps, train_data):
    eps_scheduler = eval(args.scheduler_name)(max_eps, args.scheduler_opts)
    epoch_length = int((len(train_data.dataset) + train_data.batch_size - 1) / train_data.batch_size)
    eps_scheduler.set_epoch_length(epoch_length)
    return eps_scheduler

def get_coef_scheduler(args, max_coef, train_data):
    coef_scheduler = eval(args.scheduler_name)(max_coef, args.coef_scheduler_opts)
    epoch_length = int((len(train_data.dataset) + train_data.batch_size - 1) / train_data.batch_size)
    coef_scheduler.set_epoch_length(epoch_length)
    return coef_scheduler

def get_lr_scheduler(args, opt, lr_steps=None):
    for pg in opt.param_groups:
        pg['lr'] = args.lr
    if args.lr_schedule == "multistep":
        return optim.lr_scheduler.MultiStepLR(opt,
                                              milestones=map(int, args.lr_decay_milestones.split(',')),
                                              gamma=args.lr_decay_factor)
    elif args.lr_schedule == "cyclic":
        if args.data.upper() == 'SVHN':
            scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                step_size_up=lr_steps * 2 / 5, step_size_down=lr_steps * 3 / 5)
        else:
            scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
        return scheduler


def get_optimizer(args, params, checkpoint=None):
    if args.opt == 'SGD':
        opt = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        opt = eval('optim.' + args.opt)(params, lr=args.lr, weight_decay=args.weight_decay)
    logger.info(f'Optimizer {opt}')
    if checkpoint:
        if 'optimizer' not in checkpoint:
            logger.error('Cannot find optimzier checkpoint')
        else:
            opt.load_state_dict(checkpoint['optimizer'])
    return opt


def update_relu_stat(model, meter):
    for node in model._modules.values():
        if isinstance(node, BoundRelu):
            l, u = node.inputs[0].lower, node.inputs[0].upper
            meter.update('active', (l > 0).float().sum() / l.numel())
            meter.update('inactive', (u < 0).float().sum() / l.numel())


def get_autolirpa_bns(lirpa_model):
    autolirpa_bns = []
    modules = get_modules(lirpa_model)
    for cm in modules.values():
        if isinstance(cm, BoundBatchNormalization):
            autolirpa_bns.append(cm)
    return autolirpa_bns
