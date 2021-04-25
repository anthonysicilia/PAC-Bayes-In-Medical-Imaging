import numpy as np
import torch
import copy
import torch.nn.functional as F

from typing import Optional, List

from .pbb import trunc_normal_, Gaussian, Laplace, ProbConv2d

def init_conv_weights(m, activations='relu'):

    gain = torch.nn.init.calculate_gain(activations)

    if type(m) == torch.nn.Conv2d  \
            or type(m) == torch.nn.ConvTranspose2d:

        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        torch.nn.init.constant_(m.bias, 0.0)

class ProbConvTranspose2d(torch.nn.Module):
    """
    Adapted from Prob models at the following github repo:
    https://github.com/mperezortiz/PBB
    with associated paper: 
    https://arxiv.org/abs/2007.12911
    as well as the pytorch source code repo:
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#ConvTranspose2d
    Implementation of a Probabilistic Convolutional Transpose layer.
    Some Parameters Possibly unknown to the User
    ----------
    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)
    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior
    init_layer : Layer object
        Layer object used to initialise the prior
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 rho_prior,
                 prior_dist='gaussian',
                 device='cuda',
                 stride=1,
                 padding=0,
                 output_padding=0,
                 groups=1,
                 bias=True,
                 dilation=1,
                 padding_mode='zeros',
                 init_layer=None):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.groups = 1
        self.output_padding = 0
        self.bias = True
        self.padding_mode = 'zeros'

        assert (self.groups == groups
            and self.output_padding == output_padding
            and self.bias == bias
            and self.padding_mode == padding_mode
            and init_layer is not None), \
            "Changing certain defaults is not supported yet."
        
        self.rho_prior = rho_prior
        self.device = device

        if init_layer:
            weights_mu_init = init_layer.weight
            bias_mu_init = init_layer.bias
        else:
            raise NotImplementedError('Init layer must be defined.')

        # set scale parameters, NOTE: this is transposed
        weights_rho_init = torch.ones(in_channels, out_channels,
                                      *self.kernel_size) * rho_prior
        bias_rho_init = torch.ones(out_channels) * rho_prior

        if prior_dist == 'gaussian':
            dist = Gaussian
        elif prior_dist == 'laplace':
            dist = Laplace
        else:
            raise RuntimeError(f'Wrong prior_dist {prior_dist}')
        
        self.dist_init = dist

        self.weight = dist(weights_mu_init.clone(),
                           weights_rho_init.clone(),
                           device=device,
                           fixed=False)
        self.bias = dist(bias_mu_init.clone(),
                         bias_rho_init.clone(),
                         device=device,
                         fixed=False)
        self.weight_prior = dist(weights_mu_init.clone(),
                                 weights_rho_init.clone(),
                                 device=device,
                                 fixed=True)
        self.bias_prior = dist(bias_mu_init.clone(),
                               bias_rho_init.clone(),
                               device=device,
                               fixed=True)

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
            self.kl_div = self.weight.compute_kl(
                self.weight_prior) + self.bias.compute_kl(self.bias_prior)

        return F.conv_transpose2d(input, weight, bias, self.stride,
                                  self.padding, self.output_padding, self.groups,
                                  self.dilation)
    
    def reset_prior(self, init_layer):
        weights_mu_init = init_layer.weight
        bias_mu_init = init_layer.bias
        weights_rho_init = torch.ones(self.in_channels, self.out_channels, *self.kernel_size) * self.rho_prior
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

class ProbSequential(torch.nn.Sequential):

    """
    Adapted from PyTorch source repo:
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/container.html#Sequential
    NOTE: This implementation requires manually decalring top-level prob modules!
    """

    def forward(self, input, sample=False):
        for m in self:
            isprob = isinstance(m, ProbConvTranspose2d)
            isprob = isprob or isinstance(m, ProbConv2d)
            isprob = isprob or isinstance(m, ProbUNet.Up)
            if isprob:
                input = m(input, sample=sample)
            else:
                input = m(input)
        return input

class ProbUNet(torch.nn.Module):
    """
    Adapted from deterministic UNet class below...
    as well as from the Prob models at the following github repo:
    https://github.com/mperezortiz/PBB
    with associated paper: 
    https://arxiv.org/abs/2007.12911
    """

    def __init__(self,
                 rho_prior,
                 prior_dist,
                 bilinear=False,
                 device='cpu',
                 keep_batchnorm=False,
                 init_net=None):

        super().__init__()

        assert init_net is not None

        self.input_conv = self._double_conv(3, 64, rho_prior, 
            prior_dist, init_net.input_conv, keep_batchnorm, device)
        self.down1 = self._down(64, 128, rho_prior, prior_dist, 
            init_net.down1, keep_batchnorm, device)
        self.down2 = self._down(128, 256, rho_prior, prior_dist,
            init_net.down2, keep_batchnorm, device)
        self.down3 = self._down(256, 512, rho_prior, prior_dist,
            init_net.down3, keep_batchnorm, device)
        self.down4 = self._down(512, 512, rho_prior, prior_dist,
            init_net.down4, keep_batchnorm, device)
        self.up1 = self._up(1024, 256, rho_prior, prior_dist,
            init_net.up1, keep_batchnorm, device, bilinear)
        self.up2 = self._up(512, 128, rho_prior, prior_dist, 
            init_net.up2, keep_batchnorm, device, bilinear)
        self.up3 = self._up(256, 64, rho_prior, prior_dist, 
            init_net.up3, keep_batchnorm, device, bilinear)
        self.up4 = self._up(128, 64, rho_prior, prior_dist, 
            init_net.up4, keep_batchnorm, device, bilinear)
        self.output_conv = ProbConv2d(in_channels=64,
            out_channels=1, rho_prior=rho_prior,
            prior_dist=prior_dist, device=device,
            kernel_size=1, init_layer=init_net.output_conv)
        self.device = device
    
    def forward(self, x, sample=False):
        h = self.input_conv(x, sample=sample)
        d1 = self.down1(h, sample=sample)
        d2 = self.down2(d1, sample=sample)
        d3 = self.down3(d2, sample=sample)
        d4 = self.down4(d3, sample=sample)
        u1 = self.up1((d4, d3), sample=sample)
        u2 = self.up2((u1, d2), sample=sample)
        u3 = self.up3((u2, d1), sample=sample)
        u4 = self.up4((u3, h), sample=sample)
        output = self.output_conv(u4, sample=sample)
        return output

    def _double_conv(self,
                      in_channels,
                      out_channels,
                      rho_prior,
                      prior_dist,
                      init_module,
                      keep_batchnorm,
                      device,
                      kernel_size=3):
        ic = in_channels
        oc = out_channels
        # print(init_module)
        return ProbSequential(
            ProbConv2d(in_channels=ic,
                       out_channels=oc,
                       rho_prior=rho_prior,
                       prior_dist=prior_dist,
                       device=device,
                       kernel_size=kernel_size,
                       padding=1,
                       # force affine False since we don't have a ProbBatchNorm
                       init_layer=init_module[0]), 
            torch.nn.BatchNorm2d(oc, affine=False) if not keep_batchnorm \
                else copy.deepcopy(init_module[1]),
            torch.nn.ReLU(inplace=True),
            ProbConv2d(in_channels=oc,
                       out_channels=oc,
                       rho_prior=rho_prior,
                       prior_dist=prior_dist,
                       device=device,
                       kernel_size=kernel_size,
                       padding=1,
                       # force affine False since we don't have a ProbBatchNorm
                       init_layer=init_module[3]), 
            torch.nn.BatchNorm2d(oc, affine=False) if not keep_batchnorm \
                else copy.deepcopy(init_module[4]),
            torch.nn.ReLU(inplace=True))

    def _down(self, in_channels, out_channels, rho_prior, prior_dist, init_module, 
        keep_batchnorm, device):
        # print(init_module)
        return ProbSequential(
            torch.nn.MaxPool2d(2),
            self._double_conv(in_channels, out_channels, rho_prior,
                prior_dist, init_module[1], keep_batchnorm, device))

    def _up(self,
            in_channels,
            out_channels,
            rho_prior,
            prior_dist,
            init_module,
            keep_batchnorm,
            device,
            bilinear=False):
        return ProbSequential(
            ProbUNet.Up(in_channels,
                        out_channels,
                        rho_prior,
                        prior_dist,
                        init_module[0],
                        device,
                        bilinear=bilinear),
            self._double_conv(in_channels, out_channels, rho_prior,
                prior_dist, init_module[1], keep_batchnorm, device))

    class Up(torch.nn.Module):

        def __init__(self,
                     in_channels,
                     out_channels,
                     rho_prior,
                     prior_dist,
                     init_module,
                     device,
                     bilinear=False):
            ic = in_channels
            oc = out_channels
            super().__init__()
            assert not bilinear, 'Bilinear upsample not supported in ProbUNet'
            self.up = ProbConvTranspose2d(in_channels=ic // 2,
                                            out_channels=ic // 2,
                                            kernel_size=2,
                                            rho_prior=rho_prior,
                                            prior_dist=prior_dist,
                                            device=device,
                                            stride=2,
                                            init_layer=init_module.up)

        def forward(self, x, sample=False):
            x1, x2 = x
            x1 = self.up(x1, sample=sample)
            # bxcxhxw
            h_diff = x2.size()[2] - x1.size()[2]
            w_diff = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, (w_diff // 2, w_diff - w_diff // 2, h_diff // 2,
                            h_diff - h_diff // 2))
            return torch.cat([x2, x1], dim=1)

class UNet(torch.nn.Module):
    """
    Adapted from work of github user milesial
    Available at: https://github.com/milesial/Pytorch-UNet
    Original Paper: https://arxiv.org/abs/1505.04597
    See UNET-LICENSE in license directory
    """
    def __init__(self, bilinear=False):
        super().__init__()
        
        self.input_conv = self._double_conv(3, 64)
        self.down1 = self._down(64, 128)
        self.down2 = self._down(128, 256)
        self.down3 = self._down(256, 512)
        self.down4 = self._down(512, 512)
        self.up1 = self._up(1024, 256, bilinear)
        self.up2 = self._up(512, 128, bilinear)
        self.up3 = self._up(256, 64, bilinear)
        self.up4 = self._up(128, 64, bilinear)
        self.output_conv = torch.nn.Conv2d(64, 1, kernel_size=1)
        self.apply(init_conv_weights)

    def forward(self, x):
        h = self.input_conv(x)
        d1 = self.down1(h)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1((d4, d3))
        u2 = self.up2((u1, d2))
        u3 = self.up3((u2, d1))
        u4 = self.up4((u3, h))
        output = self.output_conv(u4)
        return output

    def _double_conv(self, in_channels, out_channels, kernel_size=3):
        ic = in_channels
        oc = out_channels
        return torch.nn.Sequential(
            torch.nn.Conv2d(ic, oc, kernel_size, padding=1),
            torch.nn.BatchNorm2d(oc, affine=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(oc, oc, kernel_size, padding=1),
            torch.nn.BatchNorm2d(oc, affine=False),
            torch.nn.ReLU(inplace=True),
        )

    def _down(self, in_channels, out_channels):
        ic = in_channels
        oc = out_channels
        return torch.nn.Sequential(torch.nn.MaxPool2d(2),
                                   self._double_conv(ic, oc))

    def _up(self, in_channels, out_channels, bilinear):
        ic = in_channels
        oc = out_channels
        return torch.nn.Sequential(
            UNet.Up(in_channels, out_channels, bilinear=bilinear),
            self._double_conv(ic, oc))

    class Up(torch.nn.Module):
        def __init__(self, in_channels, out_channels, bilinear=True):

            ic = in_channels
            oc = out_channels
            super().__init__()

            # if bilinear, use the normal convolutions to reduce the number of channels
            if bilinear:
                self.up = torch.nn.Upsample(scale_factor=2,
                                            mode='bilinear',
                                            align_corners=True)
            else:
                self.up = torch.nn.ConvTranspose2d(ic // 2,
                                                   ic // 2,
                                                   kernel_size=2,
                                                   stride=2)

            # should be covered by parent module, but in case of
            # outside use, it does not hurt to initialize twice
            self.apply(init_conv_weights)

        def forward(self, x):
            x1, x2 = x
            x1 = self.up(x1)
            # bxcxhxw
            h_diff = x2.size()[2] - x1.size()[2]
            w_diff = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, (w_diff // 2, w_diff - w_diff // 2, h_diff // 2,
                            h_diff - h_diff // 2))
            return torch.cat([x2, x1], dim=1)

class LightWeight(torch.nn.Module):

    """
    A light weight (stochastic) unet implementation.
    """

    def __init__(self, num_filters=64):
        super().__init__()

        nf = num_filters

        in_channels = 3
        out_channels = 1

        self.conv_down = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, nf, 3), 
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(nf, affine=False), 
            torch.nn.Conv2d(nf, 2 * nf, 3),
            torch.nn.ReLU(inplace=True), 
            torch.nn.BatchNorm2d(2 * nf, affine=False),
            torch.nn.Conv2d(2 * nf, 2 * nf, 3), 
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(2 * nf, affine=False))

        self.conv_up = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(2 * nf, 2 * nf, 3),
            torch.nn.ReLU(inplace=True), 
            torch.nn.BatchNorm2d(2 * nf, affine=False),
            torch.nn.ConvTranspose2d(2 * nf, nf, 3),
            torch.nn.ReLU(inplace=True), 
            torch.nn.BatchNorm2d(nf, affine=False),
            torch.nn.ConvTranspose2d(nf, out_channels, 3))

        self.apply(init_conv_weights)

    def forward(self, x):
        h = self.conv_down(x)
        return self.conv_up(h)

class ProbLightWeight(torch.nn.Module):

    """
    A light weight (stochastic) unet implementation.
    """

    def __init__(self, rho_prior, prior_dist='gaussian',
        num_filters=64, init_net=None, keep_batchnorm=False, device='cpu'):
        super().__init__()

        assert init_net is not None

        nf = num_filters

        in_channels = 3
        out_channels = 1

        self.device = device

        self.conv_down = ProbSequential(
            ProbConv2d(in_channels, nf, 3, rho_prior,
                prior_dist=prior_dist, device=device,
                init_layer=init_net.conv_down[0]),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(nf, affine=False) if not keep_batchnorm \
                else copy.deepcopy(init_net.conv_down[2]), 
            ProbConv2d(nf, 2 * nf, 3, rho_prior,
                prior_dist=prior_dist, device=device, 
                init_layer=init_net.conv_down[3]),
            torch.nn.ReLU(inplace=True), 
            torch.nn.BatchNorm2d(2 * nf, affine=False) if not keep_batchnorm \
                else copy.deepcopy(init_net.conv_down[5]),
            ProbConv2d(2 * nf, 2 * nf, 3, rho_prior,
                prior_dist=prior_dist, device=device, 
                init_layer=init_net.conv_down[6]), 
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(2 * nf, affine=False) if not keep_batchnorm \
                else copy.deepcopy(init_net.conv_down[8]))

        self.conv_up = ProbSequential(
            ProbConvTranspose2d(2 * nf, 2 * nf, 3, rho_prior,
                prior_dist=prior_dist, device=device, 
                init_layer=init_net.conv_up[0]),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(2 * nf, affine=False) if not keep_batchnorm \
                else copy.deepcopy(init_net.conv_up[2]),
            ProbConvTranspose2d(2 * nf, nf, 3, rho_prior,
                prior_dist=prior_dist, device=device, 
                init_layer=init_net.conv_up[3]),
            torch.nn.ReLU(inplace=True), 
            torch.nn.BatchNorm2d(nf, affine=False) if not keep_batchnorm \
                else copy.deepcopy(init_net.conv_up[5]),
            ProbConvTranspose2d(nf, out_channels, 3, rho_prior,
                prior_dist=prior_dist, device=device, 
                init_layer=init_net.conv_up[6]))

        self.apply(init_conv_weights)

    def forward(self, x, sample=False):
        h = self.conv_down(x, sample=sample)
        return self.conv_up(h, sample=sample)