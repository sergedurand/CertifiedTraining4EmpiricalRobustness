import argparse


def add_arguments_regularizers_and_init(parser):
    parser.add_argument('--reg-obj', '--pre-obj', type=str, nargs='+', default=['relu', 'tightness'])
    parser.add_argument('--reg-lambda', '--pi', type=float, default=0.5)
    parser.add_argument('--reg', action='store_true')
    parser.add_argument('--ccibp', action="store_true")
    parser.add_argument('--ccibp_coeff', type=float, default=0)
    parser.add_argument('--mtlibp', action="store_true")
    parser.add_argument('--mtlibp_coeff', type=float, default=0)
    parser.add_argument('--expibp', action="store_true")
    parser.add_argument('--expibp_coeff', type=float, default=0)
    parser.add_argument('--sabr', action="store_true")
    parser.add_argument('--sabr_coeff', type=float, default=0)
    parser.add_argument('--l1_coeff', type=float, default=0.)
    parser.add_argument('--forwabs_coeff', type=float, default=0.)
    parser.add_argument('--forwabs_use_bn', action="store_true")
    parser.add_argument('--elle_coeff', type=float, default=0.)
    parser.add_argument('--elle_lambda_schedule', type=str, default='onoff', help='constant (ELLE), onoff (ELLE-A)')

    parser.add_argument('--decay_rate', type=float, default=0.99, help='decay rate after increasing lambda')
    parser.add_argument('--sensitivity', type=float, default=2, help='factor multiplying the standard deviation')

    parser.add_argument('--min-eps-reg', type=float, default=1e-6)
    parser.add_argument('--tol', type=float, default=0.5)
    parser.add_argument('--num-reg-epochs', type=int, default=0)
    parser.add_argument('--no-reg-dec', action='store_true')
    parser.add_argument('--manual-init', action='store_true')
    parser.add_argument('--init-method', type=str, default='ibp')
    parser.add_argument('--kaiming_init', action='store_true')
    parser.add_argument('--no-init', action='store_true', help='No initialization')
    parser.add_argument('--length', type=int)
    parser.add_argument('--bounding_algorithm', type=str, default='IBP', choices=['IBP', 'CROWN-IBP'],
                        help='Bounds to use for verified loss')

    parser.add_argument('--attack', type=str, default='pgd', choices=['pgd', 'orig-nfgsm'],
                        help='Attack to use for training')


def add_arguments_data(parser):
    parser.add_argument('--random-crop', type=int, default=2)
    parser.add_argument('--num-class', type=int, default=10)
    parser.add_argument('--no-data-aug', action='store_true')
    parser.add_argument('--test-batch-size', type=int)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--data_loader_workers', type=int, default=5)
    parser.add_argument('--datadir', type=str, default='data')
    parser.add_argument('--subset', type=float, default=1, help='Fraction of training data to use')
    parser.add_argument('--use_index', action='store_true', help='Use index for the dataset', default=False)
    parser.add_argument('--true_std', action='store_true', help='Use true std for normalization', default=False)
    parser.add_argument('--data', type=str, default=None,
                        choices=['CIFAR', 'CIFAR10', 'CIFAR100', 'SVHN', 'TinyImageNet', None]),
    parser.add_argument('--no-norm', action="store_true", default=False, help="Do not normalize the data")


def add_arguments_eps(parser):
    parser.add_argument('--eps', type=float)
    parser.add_argument('--min-eps', type=float, default=0)
    parser.add_argument('--init-eps', type=float)
    parser.add_argument('--fix-eps', action='store_true', help='No epsilon scheduling')
    parser.add_argument('--scheduler_name', type=str, default='SmoothedScheduler')
    parser.add_argument('--scheduler_opts', type=str, default='start=0,length=80')
    parser.add_argument('--train-eps-mul', type=float, default=1.0)


def add_arguments_coef_scheduler(parser):
    """
    Use a (optional) separate scheduling for mtl-ibp coef
    """
    parser.add_argument('--coef_init', type=float, default=0)
    parser.add_argument('--coef_scheduler_name', type=str, default='SmoothedScheduler')
    parser.add_argument('--coef_scheduler_opts', type=str, default=None)


def add_arguments_opt(parser):
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr-decay-factor', type=float, default=0.2)
    parser.add_argument('--lr-decay-milestones', type=str, default='10000')
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--lr-schedule', type=str, default='multistep')
    parser.add_argument('--lr-step', type=str, default='epoch')
    parser.add_argument('--opt', type=str, default='Adam')
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument('--grad-norm', type=float, default=10.0)
    parser.add_argument('--weight-decay', '--wd', type=float, default=0)
    parser.add_argument('--grad-acc-steps', type=int, default=1)
    parser.add_argument('--bn-mode', type=str, default='train')


def add_arguments_objective(parser):
    parser.add_argument('--loss', type=str, default='ce')


def override_neurips2021(args):
    # Rename for compatibility
    if args.method == 'default':
        args.method = 'vanilla'
    elif args.method == 'reg':
        args.method = 'fast'

    # Override arguments depending on `method`
    if args.method == 'reg-only' or args.method == 'fast' and args.no_init:
        args.reg = True
    elif args.method == 'fast':
        args.reg = args.manual_init = True
    elif args.method == 'manual':
        args.manual_init = True
    elif args.method == 'crown':
        args.bound_type = 'CROWN'
    elif args.method in ['pgd', 'fgsm', 'trades']:
        args.mode = 'adv'

    print('Regularizer:', args.reg)
    print('Manual initialization:', args.manual_init)

    if args.mode == 'adv':
        args.fix_eps = True
        args.scheduler_opts = 'start=1,length=0'
        args.scheduler_name = 'LinearScheduler'

    if args.length:
        if args.length == 20:
            args.scheduler_opts = 'start=2,length=20'
            args.lr_decay_milestones = '50,60'
            args.num_epochs = 70
        elif args.length == 80:
            args.scheduler_opts = 'start=2,length=80'
            args.lr_decay_milestones = '120,140'
            args.num_epochs = 160
        else:
            raise ValueError('Unknown length {}'.format(args.length))

    if args.reg:
        # Overridde the legacy num_reg_epochs 
        start, length = args.scheduler_opts.split(',')
        start = int(start.split('=')[1])
        length = int(length.split('=')[1])
        args.num_reg_epochs = start + length - 1


def add_arguments_pgd(parser):
    parser.add_argument('--train_att_n_steps', default=None, type=int, help='number of steps for the attack')
    parser.add_argument('--train_att_step_size', default=0.25, type=float,
                        help='step size for the attack (relative to epsilon)')
    parser.add_argument('--test_att_n_steps', default=None, type=int, help='number of steps for the attack')
    parser.add_argument('--test_att_step_size', default=None, type=float,
                        help='step size for the attack (relative to epsilon)')
    parser.add_argument('--attack_eps_factor', default=1.0, type=float, help='larger eps ratio for attack')
    parser.add_argument('--only_adv', action='store_true', help='Only use adversarial examples for training')
    parser.add_argument('--fixed_attack_eps', action='store_true', help='Only use adversarial examples for training')
    parser.add_argument('--unif', type=int, default=2, help="Multiplier for the range of uniform sampling "
                                                            "in N-FGSM. Default is 2, sampling in U[-2*eps, 2*eps]")


def add_arguments_extra_pass(parser):
    parser.add_argument("--first_pass", action="store_true", help="If true, compute clean loss even "
                                                                  "when doing adv or ibp training only."
                                                                  "The loss is computed but not backpropagated,"
                                                                  " however it impacts batch norm.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verify', '--infer', action='store_true', help='verification mode, do not train')
    parser.add_argument('--load', type=str, default='', help='Load reged model')
    parser.add_argument('--config', type=str, default="config/cifar_2_255.json")
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='use cpu or cuda')
    parser.add_argument('--log-interval', type=int, default=1000)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--dir', type=str, default='model')
    parser.add_argument('--model', type=str, default='cnn_7layer_bn_imagenet')
    parser.add_argument('--loss-fusion', action='store_true')
    parser.add_argument('--num-epochs', type=int, default=160)
    parser.add_argument('--auto-load', action='store_true',
                        help='Automatically load the latest checkpoint in the directory without specifying the checkpoint file')
    parser.add_argument('--method', type=str, default=None,
                        choices=['vanilla', 'fast', 'crown',
                                 'default', 'manual', 'reg', 'pgd', 'fgsm', 'trades'])
    parser.add_argument('--test-interval', type=int, default=1)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--natural', action='store_true', help='Natural training')
    parser.add_argument('--check-nan', action='store_true')
    parser.add_argument('--w-scale-cls', type=float, default=100, help='Weight scaling for the classification layer')

    parser.add_argument('--no-loss-fusion', action='store_true')
    parser.add_argument('--save-all', action='store_true', help='Save all the checkpoints')
    parser.add_argument('--model-params', type=str, default='')
    parser.add_argument('--mode', type=str, default='cert', choices=['cert', 'natural'])
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--wandb_label', default=None, type=str, help='if non-None, logs training data to wandb')
    parser.add_argument('--valid_share', default=None, type=float, help='if non-None, use a validation split '
                                                                        'of (1 - valid_share) of the training set')
    parser.add_argument('--valid_shuffle', action='store_true', help='Shuffle dataset for validation')
    parser.add_argument('--disable_train_tf32', action='store_true',
                        help='disable tf32 for training (already disabled everywhere else)')
    parser.add_argument('--wandb_options', type=str, default='offline', choices=['offline', 'online'])
    parser.add_argument('--wandb_tags', type=str, default='orig-nfgsm:nfgsm-only')
    parser.add_argument('--measure_time_only', action='store_true', help="log only the time per epoch")

    parser.add_argument('--start_idx', default=0, type=int, help='specific index to start')
    parser.add_argument('--end_idx', default=-1, type=int, help='specific index to end or -1 to do all')

    add_arguments_data(parser)
    add_arguments_eps(parser)
    add_arguments_opt(parser)
    add_arguments_objective(parser)
    add_arguments_regularizers_and_init(parser)
    add_arguments_pgd(parser)
    add_arguments_coef_scheduler(parser)
    add_arguments_extra_pass(parser)

    args = parser.parse_args()
    override_neurips2021(args)

    return args
