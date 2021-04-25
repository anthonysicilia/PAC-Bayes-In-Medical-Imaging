import torch
import torch.nn.functional as F
from .pbb import output_transform, ProbConv2d, ProbLinear
from ..utils.utils import PMIN
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
import copy
"""
Adapted from PyTorch source repo:
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

"""


def conv3x3(in_planes,
            out_planes,
            rho_prior,
            prior_dist='gaussian',
            device='cuda',
            init_net=None,
            stride=1,
            dilation=1):
    return ProbConv2d(in_channels=in_planes,
                      out_channels=out_planes,
                      kernel_size=3,
                      rho_prior=rho_prior,
                      prior_dist=prior_dist,
                      device=device,
                      stride=stride,
                      padding=dilation,
                      init_layer=init_net if init_net else None,
                      _bias=False)


class ProbSequential(torch.nn.Sequential):
    """
    Adapted from PyTorch source repo:
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/container.html#Sequential

    NOTE: This implementation requires manually decalring top-level prob modules!
    """
    def forward(self, input, sample=False):
        for m in self:
            isprob = isinstance(m, ProbConv2d)
            isprob = isprob or isinstance(m, BasicBlock)
            if isprob:
                input = m(input, sample=sample)
            else:
                input = m(input)
        return input


def conv1x1(in_planes,
            out_planes,
            rho_prior,
            prior_dist='gaussian',
            device='cuda',
            init_net=None,
            stride=1):
    return ProbConv2d(in_channels=in_planes,
                      out_channels=out_planes,
                      kernel_size=1,
                      rho_prior=rho_prior,
                      prior_dist=prior_dist,
                      device=device,
                      stride=stride,
                      init_layer=init_net if init_net else None,
                      _bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None,
                 rho_prior=None,
                 prior_dist='gaussian',
                 device='cuda',
                 init_net=None,
                 keep_batchnorm=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes,
                             planes,
                             rho_prior=rho_prior,
                             prior_dist=prior_dist,
                             device=device,
                             init_net=init_net.conv1,
                             stride=stride)
        self.bn1 = norm_layer(planes, affine=False) if not keep_batchnorm \
            else copy.deepcopy(init_net.bn1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes,
                             planes,
                             rho_prior,
                             prior_dist=prior_dist,
                             device=device,
                             init_net=init_net.conv2)
        self.bn2 = norm_layer(planes, affine=False) if not keep_batchnorm \
            else copy.deepcopy(init_net.bn2)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, sample=False):
        identity = x

        out = self.conv1(x, sample)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, sample)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x, sample=False)
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 rho_prior,
                 prior_dist='gaussian',
                 device='cuda',
                 init_net=None,
                 num_classes=7,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None,
                 keep_batchnorm=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.rho_prior = rho_prior
        self.prior_dist = prior_dist
        self.device = device
        self.keep_batchnorm = keep_batchnorm

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(
                                 replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = ProbConv2d(in_channels=3,
                                out_channels=self.inplanes,
                                kernel_size=7,
                                rho_prior=rho_prior,
                                prior_dist=prior_dist,
                                device=device,
                                stride=2,
                                padding=3,
                                init_layer=init_net.conv1,
                                _bias=False)
        self.bn1 = norm_layer(self.inplanes, affine=False) if not self.keep_batchnorm \
            else copy.deepcopy(init_net.bn1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block,
                                       64,
                                       layers[0],
                                       init_net=init_net.layer1)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       init_net=init_net.layer2)
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       init_net=init_net.layer3)
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       init_net=init_net.layer4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = ProbLinear(512 * block.expansion,
                             num_classes,
                             rho_prior=self.rho_prior,
                             prior_dist=self.prior_dist,
                             device=self.device,
                             init_layer=init_net.fc)

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    dilate=False,
                    init_net=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = ProbSequential(
                conv1x1(self.inplanes,
                        planes * block.expansion,
                        rho_prior=self.rho_prior,
                        prior_dist=self.prior_dist,
                        device=self.device,
                        init_net=init_net[0].downsample[0],
                        stride=stride),
                norm_layer(planes * block.expansion, affine=False) if not self.keep_batchnorm \
            else copy.deepcopy(init_net[0].downsample[1]),
            )
        layers = []
        layers.append(
            block(self.inplanes,
                  planes,
                  stride=stride,
                  downsample=downsample,
                  groups=self.groups,
                  base_width=self.base_width,
                  dilation=previous_dilation,
                  norm_layer=norm_layer,
                  rho_prior=self.rho_prior,
                  prior_dist=self.prior_dist,
                  device=self.device,
                  init_net=init_net[0],
                  keep_batchnorm=self.keep_batchnorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      norm_layer=norm_layer,
                      rho_prior=self.rho_prior,
                      prior_dist=self.prior_dist,
                      device=self.device,
                      init_net=init_net[i],
                      keep_batchnorm=self.keep_batchnorm))
        return ProbSequential(*layers)

    def forward(self, x, sample=False, clamping=True, pmin=PMIN):
        x = self.conv1(x, sample)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, sample)
        x = self.layer2(x, sample)
        x = self.layer3(x, sample)
        x = self.layer4(x, sample)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x, sample)

        x = output_transform(x, clamping, pmin)
        return x


def _resnet(block,
            layers,
            rho_prior,
            prior_dist='gaussian',
            device='cuda',
            init_net=None,
            keep_batchnorm=False):
    model = ResNet(block,
                   layers,
                   rho_prior=rho_prior,
                   prior_dist=prior_dist,
                   device=device,
                   init_net=init_net,
                   keep_batchnorm=keep_batchnorm)
    return model


def prob_resnet18(rho_prior,
                  prior_dist='gaussian',
                  device='cuda',
                  init_net=None,
                  keep_batchnorm=False):
    return _resnet(BasicBlock, [2, 2, 2, 2],
                   rho_prior=rho_prior,
                   prior_dist=prior_dist,
                   device=device,
                   init_net=init_net,
                   keep_batchnorm=keep_batchnorm)


def prob_resnet34(rho_prior,
                  prior_dist='gaussian',
                  device='cuda',
                  init_net=None,
                  keep_batchnorm=False):
    return _resnet(BasicBlock, [3, 4, 6, 3],
                   rho_prior=rho_prior,
                   prior_dist=prior_dist,
                   device=device,
                   init_net=init_net,
                   keep_batchnorm=keep_batchnorm)
