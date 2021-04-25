import torch

from tqdm import tqdm

from .pbb import ProbLinear, ProbConv2d
from .unet import ProbConvTranspose2d

def freeze_batchnorm(model):
    """
    From the following PyTorch discussion
    https://discuss.pytorch.org/t/how-to-freeze-bn-layers-while-training-the-rest-of-network-mean-and-var-wont-freeze/89736/11
    """
    for module in model.modules():
        # print(module)
        if isinstance(module, torch.nn.BatchNorm2d):
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.requires_grad_(False)
            module.eval()

def isprob(module):
    check = isinstance(module, ProbLinear)
    check = check or isinstance(module, ProbConv2d)
    check = check or isinstance(module, ProbConvTranspose2d)
    return check

def compute_kl(model):
    kl_div = torch.zeros(1, requires_grad=True).to(model.device)
    for m in model.modules():
        # find top level prob modules and sum.
        # multivariate normal with dialog cov
        # is a product distr of i.i.d uni normals
        # so we can just sum the kl divergences
        if isprob(m):
            kl_div += m.kl_div
    return kl_div

def reset_prior(model, init_net):
    for k1, m1 in model.named_modules():
        # find bottom level prob modules
        if isprob(m1):
            for k2, m2 in init_net.named_modules():
                if k1 == k2:
                    m1.reset_prior(m2)

def coupled_train_loop(post, prior, trainloader, prefixloader, post_trainer,
    prior_trainer, args, device='cpu', verbose=True):

    post_optim = torch.optim.SGD(post.parameters(),
        args.init_lr, momentum=args.momentum, 
        weight_decay=args.weight_decay)
    post_scheduler = torch.optim.lr_scheduler.StepLR(
        post_optim, step_size=args.lr_step, gamma=1e-1)
    
    prior_optim = torch.optim.SGD(prior.parameters(),
        args.init_lr,  momentum=args.momentum, 
        weight_decay=args.weight_decay)
    prior_scheduler = torch.optim.lr_scheduler.StepLR(
        prior_optim, step_size=args.lr_step, gamma=1e-1)
    
    for e in range(args.epochs):
        post.train()
        prior.train()
        if args.freeze_batchnorm:
            freeze_batchnorm(post)
        if e < args.prior_max_train:
            prior_trainer.increment_epoch()
        post_trainer.increment_epoch()
        # so tqdm can call len() and show progress
        loader = list(zip(prefixloader, trainloader))
        # zip uses min length but these should be ~ same size
        for prefixbatch, trainbatch in tqdm(loader):
            # prior
            x = prefixbatch['img'].to(device)
            y = prefixbatch['label'].to(device)
            if e < args.prior_max_train:
                prior.zero_grad()
                loss = prior_trainer(prior, x, y)
                loss.backward()
                prior_optim.step()
                with torch.no_grad():
                    reset_prior(post, prior)
            # posterior
            if e >= args.partial_decouple:
                x = torch.cat([x, trainbatch['img'].to(device)])
                y = torch.cat([y, trainbatch['label'].to(device)])
            post.zero_grad()
            loss = post_trainer(post, x, y)
            loss.backward()
            post_optim.step()
        if verbose:
            print('prior:', prior_trainer)
            print('post:', post_trainer)
        prior_scheduler.step()
        post_scheduler.step()

def train_loop(model, dataloader, trainer, args, device='cpu', verbose=True):

    optimizer = torch.optim.SGD(model.parameters(),
        args.init_lr,  momentum=args.momentum, 
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=1e-1)

    for _ in range(args.epochs):
        model.train()
        if args.freeze_batchnorm:
            freeze_batchnorm(model)
        _train_loop(model, dataloader, optimizer, trainer,
            device=device, verbose=verbose)
        scheduler.step()
    
    return model

def _train_loop(model, dataloader, optim, trainer, device='cpu', verbose=False):
    trainer.increment_epoch()
    for batch in tqdm(dataloader):
        x = batch['img'].to(device)
        y = batch['label'].to(device)
        model.zero_grad()
        loss = trainer(model, x, y)
        loss.backward()
        optim.step()
    if verbose:
        print(trainer)
    return trainer

def test_loop(model, dataloader, tester, device='cpu', verbose=False):
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            x = batch['img'].to(device)
            y = batch['label'].to(device)
            tester(model, x, y)
    if verbose:
        print(tester)
    return tester