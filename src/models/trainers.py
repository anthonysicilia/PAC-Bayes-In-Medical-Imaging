from ..bounds.functional import dziugaite_variational_bound, rivasplata_fquad_bound
from .utils import compute_kl
from ..utils.utils import compute_dsc, compute_01

class Trainer:

    def __init__(self):
        self.epoch = 0
    
    def get_epoch(self):
        return self.epoch
    
    def reset(self):
        raise NotImplementedError('Trainer is abstract.')
    
    def increment_epoch(self):
        self.epoch += 1
        self.reset()
    
    def __call__(self, model, x, y):
        raise NotImplementedError('Trainer is abstract.')
    
    def __str__(self):
        return f"epoch={self.epoch}"

class Classic(Trainer):

    def __init__(self, loss_fn, metrics={'01' : compute_01}):
        super().__init__()
        self.loss_fn = loss_fn
        self.metric_fns = metrics
        self.reset()
    
    def reset(self):
        self.loss = 0.
        self.counter = 0.
        self.metrics = {k : 0. for k,_ in self.metric_fns.items()}

    def __call__(self, model, x, y):
        yhat = model(x)
        loss = self.loss_fn(yhat, y)
        self.loss += loss.item()
        for k in self.metrics.keys():
            self.metrics[k] += self.metric_fns[k](yhat, y)
        self.counter += 1.
        return loss
    
    def __str__(self):
        avg_loss = self.loss / self.counter
        s = f'{super().__str__()} : loss={avg_loss:.4f}'
        for k in self.metrics.keys():
            m = self.metrics[k] / self.counter
            s += f' : {k}={m:.4f}'
        return s

class PacBayes(Trainer):

    def __init__(self, loss_fn, m, delta, bound='variational', 
        kl_dampening=1, metrics={'01' : compute_01}, wandb=None):
        """
        parameter notes:
        m is the number of data point samples (sort of)
        delta is the total confidence of the bound
        These params are fixed and independent of the actual 
        computation. As these values are fixed and independent 
        of the model prediction, they should not (theoretically) 
        effect gradients much. Instead, they may be chosen to help 
        the user heuristically evaluate the quality of the model
        as it trains. For example, one might select these as values 
        that will be used after training is finished.
        Other notes:
        We only sample ONE hypothesis during training.  
        """
        super().__init__()
        assert bound in ['variational', 'fquad', 'none'], 'Bound not supported.'
        if bound == 'variational':
            self.bound_fn = dziugaite_variational_bound
        elif bound == 'fquad':
            self.bound_fn = rivasplata_fquad_bound
        elif bound == 'none':
            # just use normal variational training
            self.bound_fn = lambda x, *args, **kwargs: x 
        self.loss_fn = loss_fn
        self.m = m
        self.delta = delta
        self.kl_dampening = kl_dampening
        self.metric_fns = metrics
        self.wandb = wandb
        self.reset()
    
    def reset(self):
        self.loss = 0.
        self.kl_div = 0.
        self.bound = 0.
        self.counter = 0.
        self.metrics = {k : 0. for k,_ in self.metric_fns.items()}

    def __call__(self, model, x, y):
        kl_div = compute_kl(model)
        dampened_kl_div = kl_div * self.kl_dampening
        yhat = model(x, sample=True)
        loss = self.loss_fn(yhat, y)
        # uses fixed m, delta
        bound = self.bound_fn(loss, dampened_kl_div, self.m, self.delta)
        self.loss += loss.item()
        self.bound += bound.item()
        self.kl_div += kl_div.item()
        for k in self.metrics.keys():
            self.metrics[k] += self.metric_fns[k](yhat, y)
        self.counter += 1.
        return bound
    
    def __str__(self):
        avg_loss = self.loss / self.counter
        avg_bound = self.bound / self.counter
        avg_kl_div = self.kl_div / self.counter
        s = f'{super().__str__()} : loss={avg_loss:.4f}' \
            f' : bound={avg_bound:.4f}' \
            f' : kl_div={avg_kl_div:.4f}'
        for k in self.metrics.keys():
            m = self.metrics[k] / self.counter
            s += f' : {k}={m:.4f}'
        if self.wandb is not None:
            self.wandb.log({'avg_loss (train)':avg_loss, 
                            'avg_bound (train)':avg_bound,
                            'avg_kl_div (train)':avg_kl_div, 
                            '01 (train)':m })
        return s