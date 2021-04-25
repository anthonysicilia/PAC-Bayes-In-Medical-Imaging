"""
This file contains code adapted from the PBB github repo: 
https://github.com/mperezortiz/PBB

with associated paper: 
https://arxiv.org/abs/2007.12911
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as td
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used works best if :math:`\text{mean}` is
    near the center of the interval.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
    
    
   
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Fill tensor with uniform values from [l, u]
        tensor.uniform_(l, u)

        # Use inverse cdf transform from normal distribution
        tensor.mul_(2)
        tensor.sub_(1)

        # Ensure that the values are strictly between -1 and 1 for erfinv
        eps = torch.finfo(tensor.dtype).eps
        tensor.clamp_(min=-(1. - eps), max=(1. - eps))
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp one last time to ensure it's still in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

class Gaussian(nn.Module):
    """Implementation of a Gaussian random variable, using softplus for
    the standard deviation and with implementation of sampling and KL
    divergence computation.
    Parameters
    ----------
    mu : Tensor of floats
        Centers of the Gaussian.
    rho : Tensor of floats
        Scale parameter of the Gaussian (to be transformed to std
        via the softplus function)
    device : string
        Device the code will run in (e.g. 'cuda')
    fixed : bool
        Boolean indicating whether the Gaussian is supposed to be fixed
        or learnt.
    """

    def __init__(self, mu, rho, device='cuda', fixed=False):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad=not fixed).to(device)
        self.rho = nn.Parameter(rho, requires_grad=not fixed).to(device)
        self.device = device

    @property
    def sigma(self):
        # Computation of standard deviation:
        # We use rho instead of sigma so that sigma is always positive during
        # the optimisation. Specifically, we use sigma = log(exp(rho)+1)
        m = nn.Softplus()
        return m(self.rho)

    def sample(self):
        # Return a sample from the Gaussian distribution
        epsilon = torch.randn(self.sigma.size()).to(self.device)
        return self.mu + self.sigma * epsilon

    def compute_kl(self, other):
        # Compute KL divergence between two Gaussians (self and other)
        # (refer to the paper)
        # b is the variance of priors
        b1 = torch.pow(self.sigma, 2)
        b0 = torch.pow(other.sigma, 2)

        term1 = torch.log(torch.div(b0, b1))
        term2 = torch.div(
            torch.pow(self.mu - other.mu, 2), b0)
        term3 = torch.div(b1, b0)
        kl_div = (torch.mul(term1 + term2 + term3 - 1, 0.5)).sum()
        return kl_div

class Laplace(nn.Module):
    """Implementation of a Laplace random variable, using softplus for
    the scale parameter and with implementation of sampling and KL
    divergence computation.
    Parameters
    ----------
    mu : Tensor of floats
        Centers of the Laplace distr.
    rho : Tensor of floats
        Scale parameter for the distribution (to be transformed
        via the softplus function)
    device : string
        Device the code will run in (e.g. 'cuda')
    fixed : bool
        Boolean indicating whether the distribution is supposed to be fixed
        or learnt.
    """

    def __init__(self, mu, rho, device='cuda', fixed=False):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad=not fixed).to(device)
        self.rho = nn.Parameter(rho, requires_grad=not fixed).to(device)
        self.device = device

    @property
    def scale(self):
        # We use rho instead of sigma so that sigma is always positive during
        # the optimisation. We use sigma = log(exp(rho)+1)
        m = nn.Softplus()
        return m(self.rho)

    def sample(self):
        # Return a sample from the Laplace distribution
        # we do scaling due to numerical issues
        epsilon = (0.999*torch.rand(self.scale.size())-0.49999).to(self.device)
        result = self.mu - torch.mul(torch.mul(self.scale, torch.sign(epsilon)),
                                     torch.log(1-2*torch.abs(epsilon)))
        return result

    def compute_kl(self, other):
        # Compute KL divergence between two Laplaces distr. (self and other)
        # (refer to the paper)
        # b is the variance of priors
        b1 = self.scale
        b0 = other.scale
        term1 = torch.log(torch.div(b0, b1))
        aux = torch.abs(self.mu - other.mu)
        term2 = torch.div(aux, b0)
        term3 = torch.div(b1, b0) * torch.exp(torch.div(-aux, b1))

        kl_div = (term1 + term2 + term3 - 1).sum()
        return kl_div

class ProbLinear(nn.Module):
    """Implementation of a Probabilistic Linear layer.
    Parameters
    ----------
    in_features : int
        Number of input features for the layer
    out_features : int
        Number of output features for the layer
    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)
    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior
    device : string
        Device the code will run in (e.g. 'cuda')
    init_layer : Linear object
        Linear layer object used to initialise the prior
    """

    def __init__(self, in_features, out_features, rho_prior, prior_dist='gaussian', device='cuda', init_layer=None):
        super().__init__()

        self.device = device

        self.in_features = in_features
        self.out_features = out_features

        # Set sigma for the truncated gaussian of weights
        sigma_weights = 1/np.sqrt(in_features)

        if init_layer:
            weights_mu_init = init_layer.weight
            bias_mu_init = init_layer.bias
        else:
            # Initialise distribution means using truncated normal
            weights_mu_init = trunc_normal_(torch.Tensor(
                out_features, in_features), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
            bias_mu_init = torch.zeros(out_features)

        weights_rho_init = torch.ones(out_features, in_features) * rho_prior
        bias_rho_init = torch.ones(out_features) * rho_prior

        self.rho_prior = rho_prior

        if prior_dist == 'gaussian':
            dist = Gaussian
        elif prior_dist == 'laplace':
            dist = Laplace
        else:
            raise RuntimeError(f'Wrong prior_dist {prior_dist}')
        
        self.dist_init = dist

        self.bias = dist(bias_mu_init.clone(),
                         bias_rho_init.clone(), device=device, fixed=False)
        self.weight = dist(weights_mu_init.clone(),
                           weights_rho_init.clone(), device=device, fixed=False)
        self.weight_prior = dist(
            weights_mu_init.clone(), weights_rho_init.clone(), device=device, fixed=True)
        self.bias_prior = dist(
            bias_mu_init.clone(), bias_rho_init.clone(), device=device, fixed=True)

        self.kl_div = 0

    def forward(self, input, sample=False):
        if self.training or sample:
            # during training we sample from the model distribution
            # sample = True can also be set during testing if we
            # want to use the stochastic/ensemble predictors
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            # otherwise we use the posterior mean
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training:
            # sum of the KL computed for weights and biases
            self.kl_div = self.weight.compute_kl(self.weight_prior) + \
                self.bias.compute_kl(self.bias_prior)

        return F.linear(input, weight, bias)
    
    def reset_prior(self, init_layer):
        weights_mu_init = init_layer.weight
        bias_mu_init = init_layer.bias
        weights_rho_init = torch.ones(self.out_features, self.in_features) * self.rho_prior
        bias_rho_init = torch.ones(self.out_features) * self.rho_prior
        self.weight_prior = self.dist_init(
            weights_mu_init.clone(), weights_rho_init.clone(), device=self.device, fixed=True)
        self.bias_prior = self.dist_init(
            bias_mu_init.clone(), bias_rho_init.clone(), device=self.device, fixed=True)
        with torch.no_grad():
            # this setting of kl div is never used for optimization
            # do this to make sure it is never used, and free up mem.
            self.kl_div = self.weight.compute_kl(self.weight_prior) + \
                self.bias.compute_kl(self.bias_prior)

class ProbConv2d(nn.Module):
    """Implementation of a Probabilistic Convolutional layer.
    Parameters
    ----------
    in_channels : int
        Number of input channels for the layer
    out_channels : int
        Number of output channels for the layer
    kernel_size : int
        size of the convolutional kernel
    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)
    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior
    device : string
        Device the code will run in (e.g. 'cuda')
    stride : int
        Stride of the convolution
    padding: int
        Zero-padding added to both sides of the input
    dilation: int
        Spacing between kernel elements
    init_layer : Linear object
        Linear layer object used to initialise the prior
    """

    def __init__(self, in_channels, out_channels, kernel_size, rho_prior, prior_dist='gaussian',
                 device='cuda', stride=1, padding=0, dilation=1, init_layer=None, _bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1

        self.rho_prior = rho_prior
        self.device = device
        self._bias = _bias

        # Compute and set sigma for the truncated gaussian of weights
        in_features = self.in_channels
        for k in self.kernel_size:
            in_features *= k
        sigma_weights = 1/np.sqrt(in_features)

        if init_layer:
            weights_mu_init = init_layer.weight
            if self._bias:
                bias_mu_init = init_layer.bias
        else:
            weights_mu_init = trunc_normal_(torch.Tensor(
                out_channels, in_channels, *self.kernel_size), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
            if self._bias:
                bias_mu_init = torch.zeros(out_channels)

        # set scale parameters
        weights_rho_init = torch.ones(
            out_channels, in_channels, *self.kernel_size) * rho_prior
        if self._bias:
            bias_rho_init = torch.ones(out_channels) * rho_prior

        if prior_dist == 'gaussian':
            dist = Gaussian
        elif prior_dist == 'laplace':
            dist = Laplace
        else:
            raise RuntimeError(f'Wrong prior_dist {prior_dist}')
        
        self.dist_init = dist

        self.weight = dist(weights_mu_init.clone(),
                           weights_rho_init.clone(), device=device, fixed=False)
        if self._bias:
            self.bias = dist(bias_mu_init.clone(),
                            bias_rho_init.clone(), device=device, fixed=False)
        self.weight_prior = dist(
            weights_mu_init.clone(), weights_rho_init.clone(), device=device, fixed=True)
        if self._bias:
            self.bias_prior = dist(
                bias_mu_init.clone(), bias_rho_init.clone(), device=device, fixed=True)

        self.kl_div = 0

    def forward(self, input, sample=False):
        if self.training or sample:
            # during training we sample from the model distribution
            # sample = True can also be set during testing if we
            # want to use the stochastic/ensemble predictors
            weight = self.weight.sample()
            if self._bias:
                bias = self.bias.sample()
            else:
                bias = None
        else:
            # otherwise we use the posterior mean
            weight = self.weight.mu
            if self._bias:
                bias = self.bias.mu
            else:
                bias = None
        if self.training:
            # sum of the KL computed for weights and biases
            self.kl_div = self.weight.compute_kl(
                self.weight_prior) + (self.bias.compute_kl(self.bias_prior) if self._bias else 0)

        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
    
    def reset_prior(self, init_layer):
        weights_mu_init = init_layer.weight
        bias_mu_init = init_layer.bias
        weights_rho_init = torch.ones(self.out_channels, self.in_channels, *self.kernel_size) * self.rho_prior
        bias_rho_init = torch.ones(self.out_channels) * self.rho_prior
        self.weight_prior = self.dist_init(
            weights_mu_init.clone(), weights_rho_init.clone(), device=self.device, fixed=True)
        self.bias_prior = self.dist_init(
            bias_mu_init.clone(), bias_rho_init.clone(), device=self.device, fixed=True)
        with torch.no_grad():
            # this setting of kl div is never used for optimization
            # do this to make sure it is never used, and free up mem.
            self.kl_div = self.weight.compute_kl(self.weight_prior) + \
                self.bias.compute_kl(self.bias_prior)

class CNNet4l(nn.Module):
    """Implementation of a standard Convolutional Neural Network with 4 layers
    (used for the experiments on MNIST so it assumes a specific input size and
    number of classes)
    Parameters
    ----------
    dropout_prob : float
        probability of an element to be zeroed.
    device : string
        Device the code will run in (e.g. 'cuda')
    """

    def __init__(self, dropout_prob=0., num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.d = nn.Dropout2d(dropout_prob)

    def forward(self, x):
        x = self.d(self.conv1(x))
        x = F.relu(x)
        x = self.d(self.conv2(x))
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.d(self.fc1(x))
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class ProbNNet4l(nn.Module):
    """Implementation of a Probabilistic Neural Network with 4 layers
    (used for the experiments on MNIST so it assumes a specific input size and
    number of classes)
    Parameters
    ----------
    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)
    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior
    device : string
        Device the code will run in (e.g. 'cuda')
    init_net : NNet object
        Network object used to initialise the prior
    """

    def __init__(self, rho_prior, prior_dist='gaussian', device='cuda', init_net=None, num_classes=10):
        super().__init__()
        self.device = device
        self.l1 = ProbLinear(28*28, 600, rho_prior, prior_dist=prior_dist,
                             device=device, init_layer=init_net.l1 if init_net else None)
        self.l2 = ProbLinear(600, 600, rho_prior, prior_dist=prior_dist,
                             device=device, init_layer=init_net.l2 if init_net else None)
        self.l3 = ProbLinear(600, 600, rho_prior, prior_dist=prior_dist,
                             device=device, init_layer=init_net.l3 if init_net else None)
        self.l4 = ProbLinear(600, num_classes, rho_prior, prior_dist=prior_dist,
                             device=device, init_layer=init_net.l4 if init_net else None)

    def forward(self, x, sample=False, clamping=True, pmin=1e-4):
        x = x.view(-1, 28*28)
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = F.relu(self.l3(x, sample))
        x = output_transform(self.l4(x, sample), clamping, pmin)
        return x

    def compute_kl(self):
        # KL as a sum of the KL for each individual layer
        return self.l1.kl_div + self.l2.kl_div + self.l3.kl_div + self.l4.kl_div


class ProbCNNet4l(nn.Module):
    """Implementation of a Probabilistic Convolutional Neural Network with 4 layers
    (used for the experiments on MNIST so it assumes a specific input size,
    number of classes and kernel size).
    Parameters
    ----------
    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)
    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior
    device : string
        Device the code will run in (e.g. 'cuda')
    init_net : CNNet object
        Network object used to initialise the prior
    """

    def __init__(self, rho_prior, prior_dist='gaussian', device='cuda', init_net=None, num_classes=10):
        super().__init__()
        self.device = device
        self.conv1 = ProbConv2d(
            1, 32, 3, rho_prior, prior_dist=prior_dist, device=device, init_layer=init_net.conv1 if init_net else None)
        self.conv2 = ProbConv2d(
            32, 64, 3, rho_prior, prior_dist=prior_dist, device=device, init_layer=init_net.conv2 if init_net else None)
        self.fc1 = ProbLinear(9216, 128, rho_prior, prior_dist=prior_dist,
                              device=device, init_layer=init_net.fc1 if init_net else None)
        self.fc2 = ProbLinear(128, num_classes, rho_prior, prior_dist=prior_dist,
                              device=device, init_layer=init_net.fc2 if init_net else None)

    def forward(self, x, sample=False, clamping=True, pmin=1e-4):
        # forward pass for the network
        x = F.relu(self.conv1(x, sample))
        x = F.relu(self.conv2(x, sample))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x, sample))
        x = output_transform(self.fc2(x, sample), clamping, pmin)
        return x

    def compute_kl(self):
        # KL as a sum of the KL for each individual layer
        return self.conv1.kl_div + self.conv2.kl_div + self.fc1.kl_div + self.fc2.kl_div


class CNNet9l(nn.Module):
    """Implementation of a Convolutional Neural Network with 9 layers
    (used for the experiments on CIFAR-10 so it assumes a specific input size,
    number of classes, architecture, kernel size, etc.).
    Parameters
    ----------
    dropout_prob : float
        probability of an element to be zeroed.
    """

    def __init__(self, dropout_prob=0., num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.fcl1 = nn.Linear(4096, 1024)
        self.fcl2 = nn.Linear(1024, 512)
        self.fcl3 = nn.Linear(512, num_classes)
        self.d = nn.Dropout2d(dropout_prob)

    def forward(self, x):
        # conv layers
        x = self.d(F.relu(self.conv1(x)))
        x = self.d(F.relu(self.conv2(x)))
        x = (F.max_pool2d(x, kernel_size=2, stride=2))
        x = self.d(F.relu(self.conv3(x)))
        x = self.d(F.relu(self.conv4(x)))
        x = (F.max_pool2d(x, kernel_size=2, stride=2))
        x = self.d(F.relu(self.conv5(x)))
        x = self.d(F.relu(self.conv6(x)))
        x = (F.max_pool2d(x, kernel_size=2, stride=2))
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = F.relu(self.d(self.fcl1(x)))
        x = F.relu(self.d(self.fcl2(x)))
        x = self.fcl3(x)
        x = F.log_softmax(x, dim=1)
        return x


class ProbCNNet9l(nn.Module):
    """Implementation of a Probabilistic Convolutional Neural Network with 9 layers
    (used for the experiments on CIFAR-10 so it assumes a specific input size,
    number of classes, architecture, kernel size, etc.).
    Parameters
    ----------
    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)
    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior
    device : string
        Device the code will run in (e.g. 'cuda')
    init_net : CNNet object
        Network object used to initialise the prior
    """

    def __init__(self, rho_prior, prior_dist, device='cuda', init_net=None, num_classes=10):
        super().__init__()
        self.device = device
        self.conv1 = ProbConv2d(
            in_channels=3, out_channels=32, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
            kernel_size=3, padding=1, init_layer=init_net.conv1 if init_net else None)
        self.conv2 = ProbConv2d(
            in_channels=32, out_channels=64, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
            kernel_size=3, padding=1, init_layer=init_net.conv2 if init_net else None)
        self.conv3 = ProbConv2d(
            in_channels=64, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
            kernel_size=3, padding=1, init_layer=init_net.conv3 if init_net else None)
        self.conv4 = ProbConv2d(
            in_channels=128, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
            kernel_size=3, padding=1, init_layer=init_net.conv4 if init_net else None)
        self.conv5 = ProbConv2d(
            in_channels=128, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
            kernel_size=3, padding=1, init_layer=init_net.conv5 if init_net else None)
        self.conv6 = ProbConv2d(
            in_channels=256, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
            kernel_size=3, padding=1, init_layer=init_net.conv6 if init_net else None)
        self.fcl1 = ProbLinear(4096, 1024, rho_prior=rho_prior,
                               prior_dist=prior_dist, device=device, init_layer=init_net.fcl1 if init_net else None)
        self.fcl2 = ProbLinear(1024, 512, rho_prior=rho_prior,
                               prior_dist=prior_dist, device=device, init_layer=init_net.fcl2 if init_net else None)
        self.fcl3 = ProbLinear(
            512, num_classes, rho_prior=rho_prior, prior_dist=prior_dist, device=device, init_layer=init_net.fcl3 if init_net else None)

    def forward(self, x, sample=False, clamping=True, pmin=1e-4):
        # conv layers
        x = F.relu(self.conv1(x, sample))
        x = F.relu(self.conv2(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x, sample))
        x = F.relu(self.conv4(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv5(x, sample))
        x = F.relu(self.conv6(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = F.relu(self.fcl1(x, sample))
        x = F.relu(self.fcl2(x, sample))
        x = self.fcl3(x, sample)
        x = output_transform(x, clamping, pmin)
        return x

    def compute_kl(self):
        # KL as a sum of the KL for each individual layer
        return self.conv1.kl_div + self.conv2.kl_div + self.conv3.kl_div + self.conv4.kl_div + self.conv5.kl_div + self.conv6.kl_div + self.fcl1.kl_div + self.fcl2.kl_div + self.fcl3.kl_div


class CNNet13l(nn.Module):
    """Implementation of a Convolutional Neural Network with 13 layers
    (used for the experiments on CIFAR-10 so it assumes a specific input size,
    number of classes, architecture, kernel size, etc.).
    Parameters
    ----------
    dropout_prob : float
        probability of an element to be zeroed.
    """

    def __init__(self, dropout_prob=0., num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.fcl1 = nn.Linear(2048, 1024)
        self.fcl2 = nn.Linear(1024, 512)
        self.fcl3 = nn.Linear(512, num_classes)
        self.d = nn.Dropout(dropout_prob)

    def forward(self, x):
        # conv layers
        x = F.relu(self.d(self.conv1(x)))
        x = F.relu(self.d(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.d(self.conv3(x)))
        x = F.relu(self.d(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.d(self.conv5(x)))
        x = F.relu(self.d(self.conv6(x)))
        x = F.relu(self.d(self.conv7(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.d(self.conv8(x)))
        x = F.relu(self.d(self.conv9(x)))
        x = F.relu(self.d(self.conv10(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = F.relu(self.d(self.fcl1(x)))
        x = F.relu(self.d(self.fcl2(x)))
        x = self.fcl3(x)
        x = F.log_softmax(x, dim=1)
        return x


class ProbCNNet13l(nn.Module):
    """Implementation of a Probabilistic Convolutional Neural Network with 13 layers
    (used for the experiments on CIFAR-10 so it assumes a specific input size,
    number of classes, architecture, kernel size, etc.).
    Parameters
    ----------
    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)
    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior
    device : string
        Device the code will run in (e.g. 'cuda')
    init_net : CNNet object
        Network object used to initialise the prior
    """

    def __init__(self, rho_prior, prior_dist, device='cuda', init_net=None, num_classes=10):
        super().__init__()
        self.device = device
        self.conv1 = ProbConv2d(in_channels=3, out_channels=32, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv1 if init_net else None)
        self.conv2 = ProbConv2d(in_channels=32, out_channels=64, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv2 if init_net else None)
        self.conv3 = ProbConv2d(in_channels=64, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv3 if init_net else None)
        self.conv4 = ProbConv2d(in_channels=128, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv4 if init_net else None)
        self.conv5 = ProbConv2d(in_channels=128, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv5 if init_net else None)
        self.conv6 = ProbConv2d(in_channels=256, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv6 if init_net else None)
        self.conv7 = ProbConv2d(in_channels=256, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv7 if init_net else None)
        self.conv8 = ProbConv2d(in_channels=256, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv8 if init_net else None)
        self.conv9 = ProbConv2d(in_channels=512, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv9 if init_net else None)
        self.conv10 = ProbConv2d(in_channels=512, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist,
                                 device=device, kernel_size=3, padding=1, init_layer=init_net.conv10 if init_net else None)
        self.fcl1 = ProbLinear(2048, 1024, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, init_layer=init_net.fcl1 if init_net else None)
        self.fcl2 = ProbLinear(1024, 512, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, init_layer=init_net.fcl2 if init_net else None)
        self.fcl3 = ProbLinear(512, num_classes, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, init_layer=init_net.fcl3 if init_net else None)

    def forward(self, x, sample=False, clamping=True, pmin=1e-4):
        # conv layers
        x = F.relu(self.conv1(x, sample))
        x = F.relu(self.conv2(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x, sample))
        x = F.relu(self.conv4(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv5(x, sample))
        x = F.relu(self.conv6(x, sample))
        x = F.relu(self.conv7(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv8(x, sample))
        x = F.relu(self.conv9(x, sample))
        x = F.relu(self.conv10(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = F.relu(self.fcl1(x, sample))
        x = F.relu(self.fcl2(x, sample))
        x = self.fcl3(x, sample)
        x = output_transform(x, clamping, pmin)
        return x

    def compute_kl(self):
        # KL as a sum of the KL for each individual layer
        return self.conv1.kl_div + self.conv2.kl_div + self.conv3.kl_div + self.conv4.kl_div + self.conv5.kl_div + self.conv6.kl_div + self.conv7.kl_div + self.conv8.kl_div + self.conv9.kl_div + self.conv10.kl_div + self.fcl1.kl_div + self.fcl2.kl_div + self.fcl3.kl_div

def output_transform(x, clamping=True, pmin=1e-4):
    """Computes the log softmax and clamps the values using the
    min probability given by pmin.
    Parameters
    ----------
    x : tensor
        output of the network
    clamping : bool
        whether to clamp the output probabilities
    pmin : float
        threshold of probabilities to clamp.
    """
    # lower bound output prob
    output = F.log_softmax(x, dim=1)
    if clamping:
        output = torch.clamp(output, np.log(pmin))
    return output