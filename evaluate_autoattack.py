import copy
import os
import torch
import wandb
import json
import re
import functools
from tqdm import tqdm

from argparser import parse_args
from utils import set_seed, prepare_model
from config import load_config
from datasets import load_data
from attack import pgd_attack
from certified import get_C
from utils import compute_perturbation
from auto_LiRPA.utils import logger
from auto_LiRPA import BoundedModule
from autoattack import AutoAttack


def report(args, pbar, tot_aa_ok, tot_ibp_verified, tot_nat_ok, tot_pgd_ok, tot_tests):
    """ Logs evaluation statistics to standard output. """
    pbar.set_description(
        'tot_tests: %d, AAev_nat_ok: %.5lf [%d/%d], AAev_pgd_ok: %.5lf [%d/%d], AAev_aa_ok: %.5lf [%d/%d], '
        'AAev_ibp_ok: %.5lf [%d/%d], ' % (
            tot_tests,
            tot_nat_ok/tot_tests, tot_nat_ok, tot_tests,
            tot_pgd_ok/tot_tests, tot_pgd_ok, tot_tests,
            tot_aa_ok/tot_tests, tot_aa_ok, tot_tests,
            tot_ibp_verified/tot_tests, tot_ibp_verified, tot_tests,
        )
    )
    # Log evaluation data onto wandb.
    if args.wandb_label is not None:
        wandb_dict = {
            f'AAev_aa_ok': tot_aa_ok/tot_tests,
            'AAev_ibp_ver_ok': tot_ibp_verified/tot_tests,
            f'AAev_nat_ok': tot_nat_ok/tot_tests,
            f'AAev_pgd_ok': tot_pgd_ok/tot_tests,
        }
        wandb.log(wandb_dict, step=tot_tests)


def main(args):

    if torch.cuda.is_available():
        # Disable the 19-bit TF32 type, which is not precise enough for verification purposes
        # see https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    config = load_config(args.config)
    bound_config = config['bound_params']
    logger.info('config: {}'.format(json.dumps(config)))

    # Set random seed. If there was a seed in the model name, override anything from args or config.
    regexp = re.search('seed_\d', args.load)
    if regexp:
        args.seed = int(regexp.group(0).split('_')[1])
    set_seed(args.seed or config['seed'])

    # Load dataset and network.
    model_ori, checkpoint, epoch, best = prepare_model(args, logger, config)
    model_ori.eval()
    batch_size = (args.batch_size or config['batch_size'])
    test_batch_size = args.test_batch_size or 1
    # NOTE: evaluation data is loaded without normalization to comply with AutoAttack
    # (see https://github.com/fra31/auto-attack/issues/13)
    data = args.data or config['data']
    dummy_input, _, test_data = load_data(
        args, data, batch_size, test_batch_size, aug=not args.no_data_aug, unnormalized_eval=True)

    name_prefix = f"AAev_{os.path.basename(args.load)}"
    # Log certification data onto wandb.
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

    # get autolirpa model to attempt quick certification before AA
    model_lirpa = BoundedModule(
        copy.deepcopy(model_ori), dummy_input, bound_opts=config['bound_params']['bound_opts'], custom_ops={},
        device=args.device)
    model_lirpa.eval()
    model_ori.to(args.device)

    data_max, data_min, std, data_mean = test_data.data_max, test_data.data_min, test_data.std, test_data.mean

    eps = args.eps or bound_config['eps']

    # As required by AutoAttack, including normalization in the model itself
    def normalizer(cmean, cstd, cx):
        mean_view = cmean.view(-1, 1, 1) if cmean.ndim == 1 else cmean
        std_view = cstd.view(-1, 1, 1) if cstd.ndim == 1 else cstd
        return (cx - mean_view) / std_view
    normalizer_cpu = functools.partial(normalizer, data_mean, std)
    normalizer_gpu = functools.partial(normalizer, data_mean.cuda(), std.cuda())
    model_with_normalization = lambda cx: model_ori(normalizer_gpu(cx))

    # AutoAttack init
    aa_adversary = AutoAttack(model_with_normalization, norm='Linf', eps=eps, version='standard')

    norm_eps = eps
    if type(norm_eps) == float:
        norm_eps = (norm_eps / std).view(1, -1, 1, 1)
    else:  # [batch_size, channels]
        norm_eps = (norm_eps.view(*norm_eps.shape, 1, 1) / std.view(1, -1, 1, 1))

    pbar = tqdm(test_data, dynamic_ncols=True)
    tot_aa_ok, tot_ibp_verified, tot_nat_ok, tot_pgd_ok, tot_tests = 0, 0, 0, 0, 0
    for test_idx, (inputs, targets) in enumerate(pbar):
        if tot_tests < args.start_idx or (args.end_idx != -1 and tot_tests >= args.end_idx):
            continue

        tot_tests += inputs.shape[0]

        # Standard accuracy.
        nat_outs = model_ori(normalizer_cpu(inputs).cuda()).cpu()
        nat_ok = targets.eq(nat_outs.max(dim=1)[1])
        vulnerable = ~nat_ok

        # Logging.
        tot_nat_ok += nat_ok.sum().item()
        if vulnerable.all():
            print("==========> All batch points misclassified")
            report(args, pbar, tot_aa_ok, tot_ibp_verified, tot_nat_ok, tot_pgd_ok, tot_tests)
            continue

        # Compute the perturbation
        data_ub = torch.min(normalizer_cpu(inputs) + norm_eps, data_max)
        data_lb = torch.max(normalizer_cpu(inputs) - norm_eps, data_min)

        # Run a quick attack before BaB
        with torch.no_grad():
            adv_data = pgd_attack(
                model_ori, data_lb.cuda(), data_ub.cuda(),
                lambda x: torch.nn.CrossEntropyLoss(reduction='none')(x, targets.cuda()),
                args.test_att_n_steps, args.test_att_step_size)
            adv_outs = model_ori(adv_data.cuda()).cpu()
            adv_ok = targets.eq(adv_outs.max(dim=1)[1])
            vulnerable = vulnerable | (~adv_ok)
        tot_pgd_ok += adv_ok.sum().item()
        if vulnerable.all():
            print("==========> All batch points either misclassified or vulnerable after a PGD attack")
            report(args, pbar, tot_aa_ok, tot_ibp_verified, tot_nat_ok, tot_pgd_ok, tot_tests)
            continue

        # Check whether IBP can verify robustness before doing AA
        c = get_C(args, normalizer_cpu(inputs).to(args.device), targets.to(args.device))
        x, data_lb, data_ub = compute_perturbation(
            args, eps, normalizer_cpu(inputs).to(args.device), data_min.to(args.device), data_max.to(args.device),
            std.to(args.device), True, False)
        ibplb, _ = model_lirpa(x=(x,), method_opt="compute_bounds", IBP=True, C=c, method=None, no_replicas=True)
        ibp_verified = (ibplb.min(dim=-1)[0] > 0).cpu()
        tot_ibp_verified += ibp_verified.sum().item()
        tot_aa_ok += ibp_verified.sum().item()  # if IBP robust, then it's AA robust too

        if (ibp_verified | vulnerable).all():
            print("==========> All batch points either misclassified/PGD-vulnerable/IBP-verified")
            report(args, pbar, tot_aa_ok, tot_ibp_verified, tot_nat_ok, tot_pgd_ok, tot_tests)
            continue

        print(f"********** Running AA on {(~(ibp_verified | vulnerable)).sum()}/{inputs.shape[0]} points")
        # Execute AA only on the points that are not misclassified/PGD-vulnerable/IBP-verified
        aa_inputs = inputs[(~(ibp_verified | vulnerable))]
        aa_targets = targets[(~(ibp_verified | vulnerable))]
        _, aa_results = aa_adversary.run_standard_evaluation(
            aa_inputs, aa_targets, bs=inputs.shape[0], return_labels=True)
        aa_ok = aa_targets.eq(aa_results)
        tot_aa_ok += aa_ok.sum().item()
        report(args, pbar, tot_aa_ok, tot_ibp_verified, tot_nat_ok, tot_pgd_ok, tot_tests)

    # Log certification data onto wandb.
    if args.wandb_label is not None:
        wandb_dict = {
            'final_AAev_aa_ok': tot_aa_ok / tot_tests,
            'final_AAev_ibp_ver_ok': tot_ibp_verified / tot_tests,
            'final_AAev_nat_ok': tot_nat_ok / tot_tests,
            'final_AAev_pgd_ok': tot_pgd_ok / tot_tests,
        }
        wandb.log(wandb_dict, step=tot_tests)

        wandb.run.summary['model_dir'] = args.load
        wandb.run.summary['host_name'] = os.uname().nodename


if __name__ == '__main__':

    args = parse_args()
    if args.wandb_options == 'offline':
        os.environ["WANDB_API_KEY"] = 'your_key'
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB__SERVICE_WAIT"] = "300"

    main(args)
