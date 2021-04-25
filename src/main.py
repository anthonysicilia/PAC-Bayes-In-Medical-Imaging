import argparse

from math import log, exp

import torch
from .models.resnet import resnet18, resnet34
from .models.prob_resnet import prob_resnet18, prob_resnet34
from .data.datasets import ISICChallengeSet
from .models.unet import UNet, ProbUNet, LightWeight, ProbLightWeight
from .models.cnn import CNNet4l, ProbCNNet4l, CNNet9l, ProbCNNet9l, \
    CNNet13l, ProbCNNet13l
from .models.trainers import Classic as ClassicTrainer
from .models.trainers import PacBayes as PacBayesTrainer
from .models.testers import PACBayes, PACBayesBound, \
    ClassicBound, Classic
from .models.utils import train_loop, test_loop, coupled_train_loop
from .utils.utils import DSCLoss, BoundedNLLLoss, set_random_seed, \
    compute_01, compute_dsc

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ################################################################
    # DATASET
    ################################################################
    parser.add_argument(
        '--task',
        type=str,
        choices=['segment', 'classify'],
        help='Task to evaluate.',
        default='classify')

    parser.add_argument(
        '--model',
        type=str,
        choices=['unet', 'light', 'cnn4l', 'cnn9l', 'cnn13l', 'resnet18','resnet34'],
        help='Model to use.',
        default='cnn9l')

    ################################################################
    # OPTIMIZATION
    ################################################################
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Batch size. For partially decoupled training, ' \
        'prior sees this many samples. Posterior sees double',
        default=64)

    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of epochs to train. When using prefix, ' \
        'prior and posterior get this number of epochs total ' \
        '(unless specified otherwise by coupling parameters).',
        default=150)

    parser.add_argument(
        '--lr_step',
        type=int,
        help='Decrease the learning rate by factor of 10 ' \
        'after this many steps.',
        default=125)

    parser.add_argument(
        '--init_lr',
        type=float,
        help='Starting value for the learning rate.',
        default=1e-3)

    parser.add_argument(
        '--weight_decay',
        type=float,
        help='Weight decay for SGD.',
        default=0.)

    parser.add_argument(
        '--momentum',
        type=float,
        help='Momentum for SGD',
        default=.9)

    ################################################################
    # PAC BAYES PARAMS
    ################################################################
    parser.add_argument(
        '--baseline',
        help='When specified, trains non-probabilistic models.',
        action='store_true')

    parser.add_argument(
        '--sigma_prior',
        type=float,
        help='Variance used in all priors.',
        default=0.01)

    parser.add_argument(
        '--prior_dist',
        type=str,
        choices=['gaussian', 'laplace'],
        help='Distribution used for prior (and posterior).',
        default='gaussian')

    parser.add_argument(
        '--delta',
        type=float,
        help='Confidence probability for all upperbounds.',
        default=0.05)

    parser.add_argument(
        '--kl_dampening',
        type=float,
        help='Dampen KL by this factor when training.',
        default=1.)

    parser.add_argument(
        '--estimator',
        type=str,
        choices=['mean', 'sample', 'ensemble'],
        help='Estimator to use for inference for prob. models.',
        default='mean')

    parser.add_argument(
        '--ensemble_samples',
        type=int,
        help='Number of samples when estimator is ensemble.',
        default=100)

    parser.add_argument(
        '--mc_samples',
        type=int,
        help='Number of hypothesis samples for prob. model bounds.',
        default=100)

    parser.add_argument(
        '--use_prefix',
        help='Use a held-out prefix dataset to optimize the prior.',
        action='store_true')

    parser.add_argument(
        '--train_bound',
        type=str,
        choices=['variational', 'fquad', 'none'],
        help='Pac Bayes bound to use for training. Selecting ' \
        '"none" uses a standard loss + model sampling.',
        default='variational')

    parser.add_argument(
        '--test_bound',
        type=str,
        choices=['mauer'],
        help='Pac Bayes bound to use when testing.',
        default='mauer')

    parser.add_argument(
        '--use_coupling',
        help='Couple posterior and prior training. This ensures ' \
        'both see the same randomness. See other constraints below.',
        action='store_true')

    parser.add_argument(
        '--partial_decouple',
        type=int,
        help='When coupling, specifies when to start introducing ' \
        'bound data to the posterior batches (starts after epoch).',
        # starts at -1; i.e., always partially decouple
        default=-1)

    parser.add_argument(
        '--prior_max_train',
        type=int,
        help='When using prefix or coupling, specifies when to ' \
        'stop training prior on prefix data (stops after epoch). ' \
        'If higher than --epochs, that value is used instead.',
        # large default; i.e., don't stop training
        default=1e4)

    ################################################################
    # UTILS
    ################################################################
    parser.add_argument(
        '--device',
        type=int,
        default=1)

    parser.add_argument(
        '--random_seed',
        type=int,
        default=0)

    parser.add_argument(
        '--num_workers',
        type=int,
        default=8)
    parser.add_argument('--wandb',
                        type=str,
                        help='Plot on wandb ',
                        default=None)

    parser.add_argument('--wandb_id',
                        type=str,
                        help='id for the current run',
                        default=None)

    parser.add_argument(
        '--freeze_batchnorm',
        help='Freezes batch norm for posterior training. This way' \
        ' we can treat batch norm like a point mass. KL div will contribute' \
        ' 0 since it is unchanged.',
        action='store_true')

    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    args = parser.parse_args()
    if args.wandb is not None:
        import wandb
        wandb.init(project=args.wandb, name=args.wandb_id)
    else:
        wandb = None

    assert not (args.baseline and args.use_prefix)
    # assert not (args.use_coupling and not args.use_prefix)

    if args.task == 'segment':
        args.task = 1
        Loss = DSCLoss
        NAME = 'dsc'
        METRIC = compute_dsc
        INVERT = True
        assert args.model in ['unet', 'light']
    elif args.task == 'classify':
        args.task = 3
        Loss = BoundedNLLLoss if not args.baseline else torch.nn.NLLLoss
        NAME = '01'
        METRIC = compute_01
        INVERT = False
        assert args.model in ['cnn4l', 'cnn9l', 'cnn13l', 'resnet18', 'resnet34']
    else:
        raise NotImplementedError(f'Unrecognized task {args.task}')

    METRICS = {NAME : METRIC}
    TESTER_ARGS = {"name" : NAME, "metric" : METRIC, "invert" : INVERT}

    set_random_seed(args.random_seed)

    use_cuda = torch.cuda.is_available()
    DEVICE = torch.device(f'cuda:{args.device}' if use_cuda else 'cpu')

    TASK_DIR = f'path/task{args.task}'

    if args.baseline:
        train = ISICChallengeSet(f'{TASK_DIR}/hoeffding_train.txt', task=args.task)
        bound = ISICChallengeSet(f'{TASK_DIR}/hoeffding_holdout.txt', task=args.task)
        test = ISICChallengeSet(f'{TASK_DIR}/final_holdout.txt', task=args.task)
    elif args.use_coupling:
        prefix = ISICChallengeSet(f'{TASK_DIR}/pac_bayes_prefix.txt', task=args.task)
        train = ISICChallengeSet(f'{TASK_DIR}/pac_bayes_prefix_bound.txt', task=args.task)
        bound = train
        test = ISICChallengeSet(f'{TASK_DIR}/final_holdout.txt', task=args.task)
    elif args.use_prefix:
        prefix = ISICChallengeSet(f'{TASK_DIR}/pac_bayes_prefix.txt', task=args.task)
        train = ISICChallengeSet(f'{TASK_DIR}/pac_bayes_full_train.txt', task=args.task)
        bound = ISICChallengeSet(f'{TASK_DIR}/pac_bayes_prefix_bound.txt', task=args.task)
        test = ISICChallengeSet(f'{TASK_DIR}/final_holdout.txt', task=args.task)
    else:
        train = ISICChallengeSet(f'{TASK_DIR}/pac_bayes_full_train.txt', task=args.task)
        bound = train
        test = ISICChallengeSet(f'{TASK_DIR}/final_holdout.txt', task=args.task)

    M = len(bound)

    LOADER_ARGS = {"batch_size" : args.batch_size,
        "num_workers" : args.num_workers,
        "shuffle" : True}

    trainloader = torch.utils.data.DataLoader(train, drop_last=True,
        **LOADER_ARGS)

    if args.use_prefix or args.use_coupling:
        prefixloader = torch.utils.data.DataLoader(prefix, drop_last=True,
            **LOADER_ARGS)
        prefixtrainer = ClassicTrainer(Loss())

    boundloader = torch.utils.data.DataLoader(bound,
        **LOADER_ARGS)

    testloader = torch.utils.data.DataLoader(test,
        **LOADER_ARGS)

    if args.model == 'unet':
        Model = UNet
        ProbModel = ProbUNet
    elif args.model == 'light':
        Model = LightWeight
        ProbModel = ProbLightWeight
    elif args.model == 'cnn4l':
        Model = CNNet4l
        ProbModel = ProbCNNet4l
    elif args.model == 'cnn9l':
        Model = CNNet9l
        ProbModel = ProbCNNet9l
    elif args.model == 'cnn13l':
        Model = CNNet13l
        ProbModel = ProbCNNet13l
    elif args.model == 'resnet18':
        Model = resnet18
        ProbModel = prob_resnet18
    elif args.model == 'resnet34':
        Model = resnet34
        ProbModel = prob_resnet34
    else:
        raise Exception(f'model {args.model} not implemented.')

    # all models starts from the same init. point
    prior = Model().to(DEVICE)
    print(prior)

    if args.use_prefix:
        # double check that coupling is not being used
        if not args.use_coupling:
            print('Selecting prior using prefix...')
            temp_e = args.epochs
            args.epochs = min(args.prior_max_train, args.epochs)
            temp_bn = args.freeze_batchnorm
            args.freeze_batchnorm = False
            train_loop(prior, prefixloader, prefixtrainer, args, device=DEVICE)
            # get the current learning rate, posterior should start at this
            num_steps = args.epochs // args.lr_step
            args.init_lr = args.init_lr * ((1e-1) ** num_steps)
            # train posterior for only the remaining number of epochs
            args.epochs = temp_e - args.epochs
            args.freeze_batchnorm = temp_bn


    # ProbModel requires explicitly passing device
    # for internal computations (in addition to casting)
    RHO_PRIOR = log(exp(args.sigma_prior) - 1.0)
    posterior = ProbModel(RHO_PRIOR, prior_dist=args.prior_dist,
        device=DEVICE, init_net=prior, keep_batchnorm=args.freeze_batchnorm).to(DEVICE)

    if args.baseline:
        # just another check
        assert not args.use_prefix
        # all models starts from the same init. point
        model = prior
    else:
        model = posterior

    if args.baseline:
        trainer = ClassicTrainer(Loss(), metrics=METRICS)
    else:
        trainer = PacBayesTrainer(Loss(), M, args.delta,
            bound=args.train_bound, metrics=METRICS, 
            kl_dampening=args.kl_dampening,wandb=wandb)

    print('Training model...')
    if args.use_coupling:
        coupled_train_loop(model, prior, trainloader, prefixloader,
            trainer, prefixtrainer, args, device=DEVICE)
    else:
        train_loop(model, trainloader, trainer, args, device=DEVICE)

    if args.baseline:
        metric_tester = Classic(name=NAME, metric=METRIC)
        bound_tester = ClassicBound(args.delta, bound='hoeffding',
            **TESTER_ARGS)
        COMPUTE_ARGS = tuple()
        BOUND_NAME = 'Hoeffding'
    else:
        metric_tester = PACBayes(estimator=args.estimator,
            ensemble_samples=args.ensemble_samples,wandb=wandb, **TESTER_ARGS)
        bound_tester = PACBayesBound(args.mc_samples, args.delta,
            bound='mauer', wandb=wandb, **TESTER_ARGS)
        COMPUTE_ARGS = (model,)
        BOUND_NAME = 'PAC Bayes (Mauer)'

    print('Computing Metrics on Final Holdout Set...')
    test_loop(model, testloader, metric_tester, device=DEVICE,
        verbose=True)
    print(f'Computing Metrics and {BOUND_NAME} Bound...')
    test_loop(model, boundloader, bound_tester,
        device=DEVICE, verbose=False)
    bound_tester.compute(*COMPUTE_ARGS)
    print(bound_tester)