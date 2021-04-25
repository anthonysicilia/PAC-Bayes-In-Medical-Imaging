import torch
import torch.nn.functional as F
from .pbb import output_transform, ProbConv2d, ProbLinear
from ..utils.utils import PMIN

"""
This file contains code adapted from the PBB github repo: 
https://github.com/mperezortiz/PBB
with associated paper: 
https://arxiv.org/abs/2007.12911
notes:
these models are similiar to those in pbb,
but we replace the last max pool layer
with an average pool layer such as below:
self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
this is similiar to many pytorch implementations, e.g. VGG
https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
Generally, this is useful since it makes the models adaptive to any input size.
The downside is that there may be signficant information loss (aggregation).
"""

NUM_CLASSES = 7

class CNNet4l(torch.nn.Module):

    def __init__(self, dropout_prob=0., num_classes=NUM_CLASSES, **kwargs):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)
        self.d = torch.nn.Dropout2d(dropout_prob)

    def forward(self, x):
        x = self.d(self.conv1(x))
        x = F.relu(x)
        x = self.d(self.conv2(x))
        x = F.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.d(self.fc1(x))
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class ProbCNNet4l(torch.nn.Module):

    def __init__(self, rho_prior, prior_dist='gaussian', device='cuda', init_net=None, num_classes=NUM_CLASSES, **kwargs):
        super().__init__()
        self.device = device
        self.conv1 = ProbConv2d(
            3, 32, 3, rho_prior, prior_dist=prior_dist, device=device, init_layer=init_net.conv1 if init_net else None)
        self.conv2 = ProbConv2d(
            32, 64, 3, rho_prior, prior_dist=prior_dist, device=device, init_layer=init_net.conv2 if init_net else None)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = ProbLinear(64 * 7 * 7, 128, rho_prior, prior_dist=prior_dist,
                              device=device, init_layer=init_net.fc1 if init_net else None)
        self.fc2 = ProbLinear(128, num_classes, rho_prior, prior_dist=prior_dist,
                              device=device, init_layer=init_net.fc2 if init_net else None)

    def forward(self, x, sample=False, clamping=True, pmin=PMIN):
        # forward pass for the network
        x = F.relu(self.conv1(x, sample))
        x = F.relu(self.conv2(x, sample))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x, sample))
        x = output_transform(self.fc2(x, sample), clamping, pmin)
        return x

    def compute_kl(self):
        # KL as a sum of the KL for each individual layer
        return self.conv1.kl_div + self.conv2.kl_div + self.fc1.kl_div + self.fc2.kl_div

class CNNet9l(torch.nn.Module):

    def __init__(self, dropout_prob=0., num_classes=NUM_CLASSES, **kwargs):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = torch.nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))
        self.fcl1 = torch.nn.Linear(256 * 7 * 7, 1024)
        self.fcl2 = torch.nn.Linear(1024, 512)
        self.fcl3 = torch.nn.Linear(512, num_classes)
        self.d = torch.nn.Dropout2d(dropout_prob)

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
        x = self.avgpool(x)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = F.relu(self.d(self.fcl1(x)))
        x = F.relu(self.d(self.fcl2(x)))
        x = self.fcl3(x)
        x = F.log_softmax(x, dim=1)
        return x


class ProbCNNet9l(torch.nn.Module):

    def __init__(self, rho_prior, prior_dist, device='cuda', init_net=None, num_classes=NUM_CLASSES, **kwargs):
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
        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))
        self.fcl1 = ProbLinear(256 * 7 * 7, 1024, rho_prior=rho_prior,
                               prior_dist=prior_dist, device=device, init_layer=init_net.fcl1 if init_net else None)
        self.fcl2 = ProbLinear(1024, 512, rho_prior=rho_prior,
                               prior_dist=prior_dist, device=device, init_layer=init_net.fcl2 if init_net else None)
        self.fcl3 = ProbLinear(
            512, num_classes, rho_prior=rho_prior, prior_dist=prior_dist, device=device, init_layer=init_net.fcl3 if init_net else None)

    def forward(self, x, sample=False, clamping=True, pmin=PMIN):
        # conv layers
        x = F.relu(self.conv1(x, sample))
        x = F.relu(self.conv2(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x, sample))
        x = F.relu(self.conv4(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv5(x, sample))
        x = F.relu(self.conv6(x, sample))
        x = self.avgpool(x)
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


class CNNet13l(torch.nn.Module):

    def __init__(self, dropout_prob=0., num_classes=NUM_CLASSES, **kwargs):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = torch.nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = torch.nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv8 = torch.nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv9 = torch.nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = torch.nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((5, 5))
        self.fcl1 = torch.nn.Linear(512 * 5 * 5, 1024)
        self.fcl2 = torch.nn.Linear(1024, 512)
        self.fcl3 = torch.nn.Linear(512, num_classes)
        self.d = torch.nn.Dropout(dropout_prob)

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
        x = self.avgpool(x)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = F.relu(self.d(self.fcl1(x)))
        x = F.relu(self.d(self.fcl2(x)))
        x = self.fcl3(x)
        x = F.log_softmax(x, dim=1)
        return x


class ProbCNNet13l(torch.nn.Module):

    def __init__(self, rho_prior, prior_dist, device='cuda', init_net=None, num_classes=NUM_CLASSES, **kwargs):
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
        self.avgpool = torch.nn.AdaptiveAvgPool2d((5, 5))
        self.fcl1 = ProbLinear(512 * 5 * 5, 1024, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, init_layer=init_net.fcl1 if init_net else None)
        self.fcl2 = ProbLinear(1024, 512, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, init_layer=init_net.fcl2 if init_net else None)
        self.fcl3 = ProbLinear(512, num_classes, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, init_layer=init_net.fcl3 if init_net else None)

    def forward(self, x, sample=False, clamping=True, pmin=PMIN):
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
        x = self.avgpool(x)
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