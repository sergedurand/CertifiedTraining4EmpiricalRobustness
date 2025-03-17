import time
import numpy as np
import torch
import math
import attack_utils
import wandb
from loguru import logger

from auto_LiRPA import BoundedModule, CrossEntropyWrapper, BoundedTensor
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import MultiAverageMeter
from auto_LiRPA.bound_ops import *
from config import load_config
from datasets import load_data, cifar10_mean, cifar10_std
from utils import *
from manual_init import manual_init, kaiming_init
from argparser import parse_args
from certified import ub_robust_loss, get_loss_over_lbs, get_C
from attack import pgd_attack, nfgsm_orig
from regularization import compute_reg, compute_L1_reg, compute_forwabs_reg, compute_elle_reg
from tqdm import tqdm
from dataclasses import dataclass, field

args = parse_args()
if not args.debug:
    logger.remove()

if not Path(args.dir).absolute().is_dir():
    assert not Path(args.dir).absolute().is_file(), f"{Path(args.dir).absolute()} is an existing file"
    Path(args.dir).absolute().mkdir(parents=True)
    print(f"Created directory {Path(args.dir).absolute()}")
if args.wandb_options == 'offline':
    import os
    os.environ["WANDB_API_KEY"] = 'your_key'
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB__SERVICE_WAIT"] = "300"


@dataclass(order=True)
class BestModel:
    metric: float
    path: Path = field(compare=False)
    epoch: int = field(compare=False)


def epsilon_clipping(eps, eps_scheduler, args, train):
    if eps < args.min_eps:
        eps = args.min_eps
    if args.fix_eps or (not train):
        eps = eps_scheduler.get_max_eps()
    if args.natural:
        eps = 0.
    return eps


def train_or_test(model, model_ori, t, loader, eps_scheduler, opt,
                  lr_scheduler=None,
                  coef_scheduler=None,
                  elle_values=None,
                  lambda_aux=None,
                  cur_lin_reg=None,
                  ):
    # Function used both for training and testing purposes

    train = opt is not None
    meter = MultiAverageMeter()

    data_max, data_min, std = loader.data_max, loader.data_min, loader.std

    if train:
        model_ori.train()
        model.train()
        eps_scheduler.train()
        eps_scheduler.step_epoch()
        if coef_scheduler is not None:
            coef_scheduler.train()
            coef_scheduler.step_epoch()
    else:
        model_ori.eval()
        model.eval()
        eps_scheduler.eval()

    pbar = tqdm(loader, dynamic_ncols=True)

    epoch_start = time.time()

    if args.elle_lambda_schedule == "onoff" and args.elle_coeff > 0.:
        lambdas = {'epoch': [], 'step': [], 'lambdas': [], 'mu': [], 'sigma': [], 'lin_err': []}

    for i, batch in enumerate(pbar):
        data, labels = batch[0], batch[1]
        start = time.time()
        eps_scheduler.step_batch()
        eps = eps_scheduler.get_eps()
        if coef_scheduler is not None:
            coef_scheduler.step_batch()

        if train:
            eps *= args.train_eps_mul
            att_n_steps = args.train_att_n_steps
            att_step_size = args.train_att_step_size
            bounding_algorithm = args.bounding_algorithm
        else:
            att_n_steps = args.test_att_n_steps
            att_step_size = args.test_att_step_size
            bounding_algorithm = "IBP"  # at eval time always use IBP

        if args.fixed_attack_eps:
            # Use a fixed epsilon only for the attack (and not for the IBP component)
            attack_eps = eps_scheduler.get_max_eps() * args.attack_eps_factor
            if train:
                attack_eps *= args.train_eps_mul
        else:
            attack_eps = eps * args.attack_eps_factor
            attack_eps = epsilon_clipping(attack_eps, eps_scheduler, args, train)

        eps = epsilon_clipping(eps, eps_scheduler, args, train)

        reg = t <= args.num_reg_epochs

        # For small eps just use natural training, no need to compute LiRPA bounds
        # NOTE: in practice, for SmoothedScheduler, if start=0, batch_method is 'robust', and robust is True
        batch_method = 'natural' if (eps < 1e-50) else 'robust'
        robust = batch_method == 'robust'

        # labels = labels.to(torch.long)
        data, labels = data.to(args.device).detach(), labels.to(args.device)

        data_batch, labels_batch = data, labels
        grad_acc = args.grad_acc_steps
        assert data.shape[0] % grad_acc == 0
        bsz = data.shape[0] // grad_acc

        for k in range(grad_acc):
            if grad_acc > 1:
                data, labels = data_batch[bsz * k:bsz * (k + 1)], labels_batch[bsz * k:bsz * (k + 1)]

            # Compute regular cross-entropy loss
            # NOTE: all forward passes should be carried out on the LiRPA model to avoid batch_norm stats mismatches
            # this will update the batch_norm stats in the LiRPA model
            if batch_method == "natural" or (not train) or args.first_pass:
                output = model(data)
                regular_ce = ce_loss(output, labels)  # regular CrossEntropyLoss used for warming up
                regular_err = torch.sum(torch.argmax(output, dim=1) != labels).item() / data.size(0)
            else:
                regular_ce = regular_err = output = None

            # Compute the perturbation
            # NOTE: at validation (train=false) these losses and errors are computed on the target epsilon
            x, data_lb, data_ub = compute_perturbation(args, eps, data, data_min, data_max, std, robust, reg)
            # Run a PGD attack
            if (att_n_steps is not None and att_n_steps > 0) or args.attack != 'pgd':

                if train:
                    # attack perturbation with a possibly different epsilon

                    _, attack_lb, attack_ub = compute_perturbation(
                        args, attack_eps, data, data_min, data_max, std, robust, reg)

                    # set the network in eval mode before the attack
                    if args.bn_mode != "train":
                        model_ori.eval()
                        model.eval()
                else:
                    attack_lb = data_lb
                    attack_ub = data_ub

                model_mode = model.training
                model_ori_mode = model_ori.training
                if wandb.run is not None and "model_mode" not in wandb.config:
                    wandb.config.update({"model_mode": model_mode, "model_ori_mode": model_ori_mode})

                logger.debug("X batch size: {}, data_lb shape: {}, data_ub shape: {}".format(
                    data.shape, data_lb.shape if data_lb is not None else 'none',
                    data_ub.shape if data_ub is not None else 'none'))
                attack = args.attack if train else "pgd"
                if attack == "pgd":
                    with torch.no_grad():
                        adv_data = pgd_attack(
                            model, attack_lb, attack_ub,
                            lambda x: nn.CrossEntropyLoss(reduction='none')(x, labels), att_n_steps, att_step_size,
                        )
                        del attack_lb, attack_ub  # save a bit of memory

                elif attack == "orig-nfgsm":
                    k = args.unif
                    # NOTE: k = 0 implies standard FGSM
                    adv_data = nfgsm_orig(
                        model=model,
                        X=data,
                        y=labels,
                        std=loader.std,
                        data_max=loader.data_max,
                        data_min=loader.data_min,
                        epsilon=attack_eps,
                        alpha=attack_eps,
                        k=k,
                    )

                if train:
                    # reset the network in train mode post-attack (the adversarial point is evaluated in train mode)
                    model_ori.train()
                    model.train()

                adv_output = model(adv_data)
                adv_loss = ce_loss(adv_output, labels)
                adv_err = torch.sum(torch.argmax(adv_output, dim=1) != labels).item() / data.size(0)

            else:
                adv_loss = regular_ce
                adv_err = regular_err
                adv_output = output

            # Upper bound on the robust loss (via IBP)
            # NOTE: when training, the bounding computation will use the BN statistics from the last forward pass: in
            # this case, from the adversarial points
            if robust or reg:
                if args.only_adv and train:
                    robust_loss = None
                    robust_err = None
                    lb = None
                elif (not args.sabr) or (not train):
                    robust_loss, robust_err, lb = ub_robust_loss(
                        args, model, x, data, labels, meter=meter, bounding_algorithm=bounding_algorithm)
                else:
                    sabr_x, sabr_center = compute_sabr_perturbation(
                        args, attack_eps, data, adv_data, data_min, data_max, std, robust, reg, coef_scheduler)
                    robust_loss, robust_err, lb = ub_robust_loss(
                        args, model, sabr_x, sabr_center, labels, meter=meter, bounding_algorithm=bounding_algorithm)

            else:
                lb = robust_loss = robust_err = None

            update_meter(meter, regular_ce, robust_loss, adv_loss, regular_err, robust_err, adv_err, data.size(0))

            if train:

                _, alpha = find_prefix(args)
                if coef_scheduler is not None:
                    alpha = coef_scheduler.get_eps()

                if reg and args.reg_lambda > 0:
                    loss = compute_reg(args, model, meter, eps, eps_scheduler)
                else:
                    loss = torch.tensor(0.).to(args.device)
                if args.l1_coeff > 0.:
                    loss += compute_L1_reg(args, model_ori, meter)
                if args.forwabs_coeff > 0.:
                    autolirpa_bns = get_autolirpa_bns(model) if args.forwabs_use_bn else None
                    loss += compute_forwabs_reg(
                        data_lb, data_ub, model_ori, alpha, use_bn=args.forwabs_use_bn, lirpa_bns=autolirpa_bns)
                if args.elle_coeff > 0.:
                    lin_err = compute_elle_reg(args, data, labels, data_lb, data_ub, model)
                    loss += cur_lin_reg * lin_err  # update loss first then compute next coef for onoff
                    if args.elle_lambda_schedule == "onoff":
                        lambdas['epoch'].append(t)
                        lambdas['step'].append(i)
                        lambdas['lambdas'].append(cur_lin_reg)
                        lambdas['mu'].append(np.mean(elle_values))
                        lambdas['sigma'].append(np.std(elle_values))
                        lambdas['lin_err'].append(lin_err.item())
                        if len(elle_values) > 2 and (
                                lin_err > np.mean(elle_values) + args.sensitivity * np.std(elle_values)):
                            # highly non linear behaviour, use max lambda
                            cur_lin_reg = lambda_aux
                        else:
                            # kinda linear behaviour, decrease lambda
                            cur_lin_reg *= args.decay_rate
                        elle_values.append(lin_err.item())

                if not (args.ccibp or args.mtlibp or args.sabr or args.expibp):

                    if robust:
                        if args.only_adv:
                            # pure adversarial training
                            loss += adv_loss
                        else:
                            # pure IBP training
                            loss += robust_loss
                    else:
                        # warmup phase
                        loss += regular_ce

                else:

                    if robust:
                        if args.ccibp:
                            # cross_entropy of convex combination of IBP with natural/adversarial logits
                            adv_diff = torch.bmm(
                                get_C(args, data, labels),
                                adv_output.unsqueeze(-1)).squeeze(-1)
                            ccibp_diff = alpha * lb + (1 - alpha) * adv_diff
                            loss += get_loss_over_lbs(ccibp_diff)
                        elif args.mtlibp:
                            if args.wandb_label is not None:
                                wandb.run.summary["coef"] = alpha
                            mtlibp_loss = alpha * robust_loss + (1 - alpha) * adv_loss
                            loss += mtlibp_loss
                        elif args.expibp:
                            expibp_loss = robust_loss ** alpha * adv_loss ** (1 - alpha)
                            loss += expibp_loss
                        else:
                            # sabr
                            sabr_loss = robust_loss
                            loss += sabr_loss
                    else:
                        # warmup phase
                        loss += regular_ce

                meter.update('Loss', loss.item(), data.size(0))
                if args.elle_coeff > 0:
                    meter.update('lin_err', lin_err.item(), data.size(0))

                loss /= grad_acc

                loss.backward()


        if train:
            grad_norm = torch.nn.utils.clip_grad_norm_(model_ori.parameters(), max_norm=args.grad_norm)
            meter.update('grad_norm', grad_norm)

            opt.step()
            opt.zero_grad()
            if args.lr_step == "batch":
                assert lr_scheduler is not None, "Learning rate scheduler must be provided in train_or_test to be \n" \
                                                 "used in batch mode"
                lr_scheduler.step()

        meter.update('Time', time.time() - start)

        pbar.set_description(
            ('[T]' if train else '[V]') +
            ' epoch=%d, adv_ok=%.4f, adv_loss=%.4f, ver_ok=%.4f, ver_loss=%.3e, eps=%.6f' % (
                t,
                1. - meter.avg('Adv_Err'),
                meter.avg('Adv_Loss'),
                1. - meter.avg('Rob_Err'),
                meter.avg('Rob_Loss'),
                eps
            )
        )

    epoch_time = time.time() - epoch_start

    if batch_method != 'natural':
        meter.update('eps', eps)

    if args.measure_time_only:
        mode = 'train' if train else 'val'
        log_dict = {f'{mode}_epoch_runtime': epoch_time}
    else:
        if train:
            log_dict = {
                'train_nat_loss': meter.avg('CE'),
                'train_nat_ok': 1. - meter.avg('Err'),
                'train_adv_ok': 1. - meter.avg('Adv_Err'),
                'train_adv_loss': meter.avg('Adv_Loss'),
                'train_ver_ok': 1. - meter.avg('Rob_Err'),
                'train_ver_loss': meter.avg('Rob_Loss'),
                'train_lr': opt.param_groups[0]['lr'],
            }
            if args.elle_coeff > 0:
                log_dict["train_lin_err"] = meter.avg("lin_err")
        else:
            log_dict = {
                'val_nat_loss': meter.avg('CE'),
                'val_nat_ok': 1. - meter.avg('Err'),
                'val_adv_ok': 1. - meter.avg('Adv_Err'),
                'val_adv_loss': meter.avg('Adv_Loss'),
                'val_ver_ok': 1. - meter.avg('Rob_Err'),
                'val_ver_loss': meter.avg('Rob_Loss'),
            }

    if args.wandb_label is not None:

        # for wandb we can log all the dict but need to clean the metrics
        new_dict = {}
        for metric, value in log_dict.items():
            mode = metric.split('_')[0]  # train or val
            if mode not in new_dict:
                new_dict[mode] = dict()
            metric = metric.split('_')[1:]  # the metric
            if len(metric) == 1:
                new_dict[mode][metric[0]] = value
            elif len(metric) == 2:
                adv_mode = metric[0]
                metric = metric[1]
                if adv_mode not in new_dict[mode]:
                    new_dict[mode][adv_mode] = dict()
                new_dict[mode][adv_mode][metric] = value
        wandb.log(new_dict, step=t)

    return meter, log_dict


def find_prefix(args):
    if args.sabr:
        ibp_method = "sabr"
        coef = float(args.sabr_coeff)
    elif args.mtlibp:
        ibp_method = "mtlib"
        coef = float(args.mtlibp_coeff)
    elif args.ccibp:
        ibp_method = "ccibp"
        coef = float(args.ccibp_coeff)
    elif args.expibp:
        ibp_method = "expibp"
        coef = float(args.expibp_coeff)
    elif args.forwabs_coeff > 0.:
        ibp_method = "forwabs"
        coef = float(args.forwabs_coeff)
    else:
        ibp_method = "adv"
        coef = 0.0
    return ibp_method, coef


def main(args):
    if torch.cuda.is_available() and args.disable_train_tf32:
        # Disable the 19-bit TF32 type, which is not precise enough for verification purposes, and seems to hurt
        # performance a bit for training
        # see https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    config = load_config(args.config)
    if args.data is not None:
        config['data'] = args.data
    else:
        args.data = config['data']  # still necessary to create lr scheduler
    torch.autograd.set_detect_anomaly(True)
    # logger.info('config: {}'.format(json.dumps(config)))
    set_seed(args.seed or config['seed'])

    model_ori, checkpoint, epoch, _ = prepare_model(args, logger, config)
    timestamp = int(time.perf_counter_ns())

    log_dict = {}
    custom_ops = {}
    bound_config = config['bound_params']
    batch_size = (args.batch_size or config['batch_size'])
    test_batch_size = args.test_batch_size or batch_size
    final_eps = args.eps or bound_config['eps']
    ibp_method, coef = find_prefix(args)
    name_prefix = f"{args.model}_{args.method if args.method is not None else ''}_{ibp_method}_coeff_{coef}_eps_{final_eps:.6f}_{timestamp}_seed_{args.seed}_attack_{args.attack}"
    # Log training data onto wandb for monitoring purposes.
    if args.wandb_label is not None:
        wandb_settings = {
            "project": "your-project",
            "entity": "your-entity",
            "group": args.wandb_label,
            "config": args,
            "name": name_prefix,
            "settings": wandb.Settings(code_dir=".", _service_wait=300),
            "tags": args.wandb_tags.split(":"),
        }
        if args.wandb_options == 'offline':
            wandb_settings["mode"] = "offline"
            wandb_settings["save_code"] = True
        wandb.init(**wandb_settings)

    # save params:
    hparams = {k: v for k, v in vars(args).items() if v is not None}
    hparams["eps"] = args.eps or bound_config['eps']

    # change list into tensor:
    for k, v in hparams.items():
        if isinstance(v, list):
            if isinstance(v[0], float):
                hparams[k] = torch.tensor(v)
            elif isinstance(v[0], str):
                hparams[k] = f"{'_'.join(v)}"

    dummy_input, train_data, test_data = load_data(
        args, config['data'], batch_size, test_batch_size, aug=not args.no_data_aug,
        use_index=args.use_index)
    bound_opts = bound_config['bound_opts']

    model_ori.eval()
    model = BoundedModule(model_ori, dummy_input, bound_opts=bound_opts, custom_ops=custom_ops, device=args.device)
    model_ori.to(args.device)

    if checkpoint is None:
        if args.manual_init:
            manual_init(args, model_ori, model, train_data)
        if args.kaiming_init:
            kaiming_init(model_ori)

    model_loss = model
    params = list(model_ori.parameters())

    opt = get_optimizer(args, params, checkpoint)
    max_eps = args.eps or bound_config['eps']
    eps_scheduler = get_eps_scheduler(args, max_eps, train_data)
    if eps_scheduler.params["start"] == 0 and eps_scheduler.params["length"] == 0:
        eps_scheduler = FixedScheduler(max_eps)
    if args.coef_scheduler_opts is not None:
        coef_scheduler = get_coef_scheduler(args, max_coef=coef, train_data=train_data)
    else:
        coef_scheduler = None
    # lr steps is epoch * num_batches
    lr_steps = args.num_epochs * len(train_data)
    lr_scheduler = get_lr_scheduler(args, opt, lr_steps=lr_steps)

    if epoch > 0 and not args.plot:
        # skip epochs
        eps_scheduler.train()
        for i in range(epoch):
            # FIXME Can use `last_epoch` argument of lr_scheduler
            if args.lr_step == "epoch":
                lr_scheduler.step()
            else:
                for batch in train_data:
                    lr_scheduler.step()
            eps_scheduler.step_epoch(verbose=False)

    if args.verify:
        start_time = time.time()
        meter, log_dict = train_or_test(model, model_ori, 10000, test_data, eps_scheduler, None,
                                        )
        logger.info(meter)
        timer = time.time() - start_time

        for k, v in log_dict.items():
            print(f"{k}: {v}")

    else:
        timer = 0.0
        best_rob = list()
        if args.elle_coeff > 0.:
            # initial values cf l:124 in ELLE/train.py
            elle_values = []
            lambda_aux = args.elle_coeff
            cur_lin_reg = args.elle_coeff if args.elle_lambda_schedule == "constant" else 0

        for t in range(epoch + 1, args.num_epochs + 1):
            start_time = time.time()
            train_or_test(model, model_ori, t, train_data, eps_scheduler, opt,
                          lr_scheduler=lr_scheduler,
                          coef_scheduler=coef_scheduler,
                          cur_lin_reg=cur_lin_reg if args.elle_coeff > 0. else None,
                          elle_values=elle_values if args.elle_coeff > 0. else None,
                          lambda_aux=lambda_aux if args.elle_coeff > 0. else None,
                          )
            update_state_dict(model_ori, model_loss)
            epoch_time = time.time() - start_time
            timer += epoch_time
            if args.lr_step == "epoch":  # if the update is done after each batch we do it inside train_or_test
                lr_scheduler.step()
            if t % args.test_interval == 0:
                # Validation phase (performed on the target epsilon)
                with torch.no_grad():
                    meter, log_dict = train_or_test(model, model_ori, t, test_data, eps_scheduler, None)

        if args.measure_time_only:
            return

        save(args, name_prefix, epoch=t, model=model_ori, opt=opt)

        # final eval: use pgd 50
        model.eval()
        with torch.no_grad():
            attacker = attack_utils.AttackUtils(
                lower_limit=test_data.data_min,
                upper_limit=test_data.data_max,
                std=test_data.std.view(-1, 1, 1),
                device=args.device
            )
            eps_255 = int(round(args.eps * 255, 0))
            with torch.enable_grad():
                pgd_loss, rob_acc = attacker.evaluate_pgd(test_data, model_ori, 50, 1, epsilon=eps_255)
                test_loss, test_acc = attacker.evaluate_standard(test_data, model)
            print(f"Final accuracy: {test_acc:.4f}, Final robust accuracy: {rob_acc:.4f}")
            if args.wandb_label is not None:
                final_eval_dict = {"final_acc": test_acc, "final_pgd_50_acc": rob_acc}
                for key in final_eval_dict:
                    wandb.run.summary[key] = final_eval_dict[key]
                wandb.config.update(final_eval_dict)  # for consistency with my upload of original nfgsm logs
                wandb.log(final_eval_dict, step=t)

    # Log training data onto wandb for monitoring purposes.
    if args.wandb_label is not None:
        wandb.run.summary["runtime"] = timer
        wandb.run.summary["model_dir"] = os.path.join(args.dir, name_prefix) + "ckpt_last"
        wandb.run.summary["host_name"] = os.uname().nodename
        if not args.verify:
            if len(best_rob) > 0:
                best_ver_acc = best_rob[0].metric
            else:
                best_ver_acc = 0.0
            wandb.run.config.update(hparams,
                                    allow_val_change=True)

            wandb.run.summary.update({
                "best_ver_acc": best_ver_acc,
                "val_nat_ok": log_dict.get("val_nat_ok", 0.0),
                "val_adv_ok": log_dict.get("val_adv_ok", 0.0),
            })
        else:
            for metric, value in log_dict.items():
                print(f"{metric}: {value}")

    # location of the saved model (printed and saved to file)
    saved_model_dir = os.path.join(args.dir, name_prefix) + "ckpt_last"
    logger.info(f"Trained model checkpoint: {saved_model_dir}")
    with open("./trained_models_info.txt", "a") as file:
        string_summary = f"Model={saved_model_dir}, Dataset={args.config}"
        for k in log_dict:
            string_summary += f", {k}={log_dict[k]}"
        file.write(string_summary + "\n")


if __name__ == '__main__':
    start = time.perf_counter()
    main(args)
    elapsed = time.perf_counter() - start
    print(f"Elapsed time: {elapsed:0.4f} seconds")
