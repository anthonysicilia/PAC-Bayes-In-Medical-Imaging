import torch
from ..bounds.functional import mauer_bound, \
    pac_bayes_hoeffding_bound, hoeffding_bound, mauer_bound_ortiz_impl
from .utils import compute_kl
from ..utils.utils import compute_dsc, compute_01

class Classic:

    def __init__(self, name='01', metric=compute_01):
        self.name = name
        self.metric_fn = metric
        self.metric = 0.
        self.counter = 0.
    
    def __call__(self, model, x, y):
        yhat = model(x)
        self.metric += self.metric_fn(yhat, y, average=False)
        # bounds will use counter for sample size
        self.counter += y.size(0)
    
    def __str__(self):
        avg_metric = self.metric / self.counter
        return f'{self.name}={avg_metric:.4f}'

class ClassicBound(Classic):

    def __init__(self, delta, name='01', metric=compute_01, bound='hoeffding', 
        invert=False):
        super().__init__(name=name, metric=metric)
        assert bound in ['hoeffding'], 'Bound not supported.'
        if bound == 'hoeffding':
            self.bound_fn = hoeffding_bound
        self.bound = None
        self.delta = delta
        self.invert = invert
    
    def compute(self):
        avg_metric = self.metric / self.counter
        if self.invert:
            avg_metric = 1 - avg_metric
        self.bound = self.bound_fn(avg_metric, self.counter, 
            self.delta)
    
    def __str__(self):
        assert self.bound is not None, "Must call compute before printing."
        s = super().__str__()
        name = self.name
        if self.invert:
            name = f'1-{name}'
        return s + f' : {name}<={self.bound:.4f}' 

class PACBayes:

    def __init__(self, estimator='mean', ensemble_samples=100, name='01', 
        metric=compute_01, invert=False,wandb=None):
        self.metric = 0.
        self.counter = 0.
        self.name = name
        self.invert = invert
        self.metric_fn = metric
        self.n = ensemble_samples
        assert estimator in ['mean', 'sample', 'ensemble']
        self.estimator = estimator
        self.wandb=wandb
    
    def __call__(self, model, x, y):
        if self.estimator == 'mean':
            yhat = model(x)
        elif self.estimator == 'sample':
            yhat = model(x, sample=True)
        elif self.estimator == 'ensemble':
            yhat = torch.cat([torch.zeros_like(x) 
                for _ in range(self.n)], dim=0)
            for i in range(self.n):
                yhat[i] = model(x, sample=True)
            yhat = torch.mean(yhat, dim=0)
        self.metric += self.metric_fn(yhat, y, average=False)
        self.counter += y.size(0)
    
    def __str__(self):
        avg_metric = self.metric / self.counter
        if self.wandb is not None:
            self.wandb.log({f'{self.name} (metric_tester)':avg_metric})
        return f'{self.name}={avg_metric:.4f}'

class PACBayesBound:

    def __init__(self, mc_samples, delta, bound='mauer', name='01', 
        metric=compute_01, invert=False, wandb=None):
        """
        parameter notes:
        mc_samples is the number of hypothesis samples to use
        delta is the total confidence of the bound
        """
        self.n = mc_samples
        self.name = name
        self.metric_fn = metric
        self.invert = invert
        self.metric = 0.
        self.counter = 0.
        self.delta = delta
        self.bound = None
        self.kl_div = None
        self.wandb = wandb
        self.set_new_bound_fn(bound)
    
    def set_new_bound_fn(self, bound):
        assert bound in ['mauer', 'hoeffding', 'ortiz'], 'Bound not supported.'
        if bound == 'mauer':
            self.bound_fn = mauer_bound
        elif bound == 'hoeffding':
            self.bound_fn = pac_bayes_hoeffding_bound
        elif bound == 'ortiz':
            # a test case to check out implementation
            self.bound_fn = mauer_bound_ortiz_impl
    
    def __call__(self, model, x, y):
        metric = 0.
        for _ in range(self.n):
            yhat = model(x, sample=True)
            metric += self.metric_fn(yhat, y, average=False)
        self.metric += metric / self.n
        self.counter += y.size(0)
    
    def compute(self, model):
        kl_div = compute_kl(model)
        self.kl_div = kl_div.item()
        avg_metric = self.metric / self.counter
        if self.invert:
            avg_metric = 1 - avg_metric
        self.bound = self.bound_fn(avg_metric, self.counter,
            self.n, self.delta, kl_div)
    
    def __str__(self):
        assert self.bound is not None, "Must call compute before printing."
        avg_metric = self.metric / self.counter
        s = f'{self.name}={avg_metric:.4f}'
        name = self.name
        if self.invert:
            name = f'1-{name}'
        if self.wandb is not None:
            self.wandb.log({f'{self.name} (bound_tester)':avg_metric,f'{name}<= (bound_tester)':self.bound,'kl_div (bound_tester)':self.kl_div})
        return s + f' : {name}<={self.bound:.4f}'