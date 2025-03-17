# File containing definitions for adversarial attacks
import random

import torch
import torch.nn.functional as F
import numpy as np
from typing import Callable


def pgd_attack(model, input_lb, input_ub, loss_function: Callable, n_steps, step_size):
    # Note that loss_function is assumed to return an entry per output coordinate (so no reduction e.g., mean)

    # model = deepcopy(model)
    device = next(model.parameters()).device
    input_lb, input_ub = input_lb.to(device), input_ub.to(device)
    step_size_scaling = (input_ub - input_lb) / 2
    attack_point = input_lb.clone().detach()
    attack_loss = (-np.inf) * torch.ones(input_lb.shape[0], dtype=torch.float32, device=input_lb.device)

    with torch.enable_grad():

        # Sample uniformly in input domain
        adv_input = (torch.zeros_like(input_lb).uniform_() * (input_ub - input_lb) + input_lb).detach_()

        for i in range(n_steps):

            adv_input.requires_grad = True
            if adv_input.grad is not None:
                adv_input.grad.zero_()

            adv_outs = model(adv_input)
            obj = loss_function(adv_outs)

            attack_point = torch.where(
                (obj >= attack_loss).view((-1,) + (1,) * (input_lb.dim() - 1)),
                adv_input.detach().clone(), attack_point)
            attack_loss = torch.where(obj >= attack_loss, obj.detach().clone(), attack_loss)

            grad = torch.autograd.grad(obj.sum(), adv_input)[0]
            adv_input = adv_input.detach() + step_size * step_size_scaling * grad.sign()
            adv_input = torch.max(torch.min(adv_input, input_ub), input_lb).detach_()

    if n_steps > 1:
        adv_outs = model(adv_input)
        obj = loss_function(adv_outs)
        attack_point = torch.where(
            (obj >= attack_loss).view((-1,) + (1,) * (input_lb.dim() - 1)),
            adv_input.detach().clone(), attack_point)
    else:
        attack_point = adv_input.detach().clone()
    torch.cuda.empty_cache()

    return attack_point

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def nfgsm_orig(model, X, y, std, data_max, data_min, epsilon, alpha, k=2):
    """
    From N-FGSM code.
    see https://github.com/pdejorge/N-FGSM/blob/main/train.py#L217
    NOTE: k = 0 implies standard FGSM, if epsilon=alpha
    """
    std, data_max, data_min = std.to(X.device), data_max.to(X.device), data_min.to(X.device)
    std = std.view(-1, 1, 1)
    epsilon /= std
    alpha /= std
    eta = torch.zeros_like(X).to(X.device)
    if k > 0:
        for j in range(len(epsilon)):
            eta[:, j, :, :].uniform_(-k * epsilon[j][0][0].item(), k * epsilon[j][0][0].item())
        eta = clamp(eta, data_min - X, data_max - X)  # - X since the noise will be added to X
    eta.requires_grad = True
    output = model(X + eta)
    loss = F.cross_entropy(output, y)
    grad = torch.autograd.grad(loss, eta)[0]
    grad = grad.detach()
    # Compute perturbation based on sign of gradient
    delta = eta + alpha * torch.sign(grad)
    delta = clamp(delta, data_min - X, data_max - X)
    delta = delta.detach()
    adv_input = X + delta

    return adv_input
