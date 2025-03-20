import random
import WeDIV2
import torch.nn as nn
import torch
import numpy as np

device = torch.device("cuda:0")

def GDnet(name, **kwargs): # Create an instance of GDnetIP model
    model = GDnetIP(vgg_structure(cfgs[name]), class_num=20)
    return model

class GDnetIP(nn.Module): # Define the network structure of GDnetIP
    def __init__(self, feature_net, class_num=20):
        super(GDnetIP, self).__init__()
        self.feature_net = feature_net
        self.inception = Inception(in_channel=256) # Inception layer 1
        self.bn1 = nn.Sequential(nn.PReLU(), nn.BatchNorm2d(256)) # Incorporating PreLU and BatchNorm (BN1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2) # Pooling layer 1
        self.inception2 = Inception(in_channel=256) # Inception 2
        self.bn2 = nn.Sequential(nn.PReLU(), nn.BatchNorm2d(256)) # BN 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1) # Pooling layer 2
        self.inception3 = Inception(in_channel=256) # Inception layer 3
        self.bn3 = nn.Sequential(nn.PReLU(), nn.BatchNorm2d(256)) # BN 3
        self.av = nn.AdaptiveAvgPool2d(1) # Global average pooling
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=1))
        self._initialize_wight()

    # For clustering-based GDnet-IP (default): inception_prob = [1.0,1.0,1.0], is_wediv = 1
    # For inception-based GDnet-IP: [prob, prob, prob], is_wediv = 0
    def forward(self, x, inception_prob=[1.0,1.0,1.0], is_wediv=1):
        x = self.feature_net(x)
        x = self.inception(x, inception_prob[0])
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.inception2(x,inception_prob[1])
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.inception3(x,inception_prob[2])
        x = self.bn3(x)
        x = self.av(x)
        if is_wediv==1:
            x = self.wediv(x)
        x = self.classifier(x)
        return x

    def _initialize_wight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)


class Inception(nn.Module):  # Define the Inception module
    def __init__(self, in_channel):
        super(Inception, self).__init__()
        self.branch1 = nn.Conv2d(in_channel, out_channels=64, kernel_size=1)
        self.branch2 = nn.Sequential(nn.Conv2d(in_channel, out_channels=64, kernel_size=1),
                                     nn.Conv2d(64, out_channels=96, kernel_size=3,
                                               padding=1))
        self.branch3 = nn.Sequential(nn.Conv2d(in_channel, out_channels=16, kernel_size=1),
                                     nn.Conv2d(16, out_channels=64, kernel_size=3,
                                               padding=1))
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                     nn.Conv2d(in_channel, out_channels=32, kernel_size=1))

    def forward(self, x, inception_prob):
        self.bernouli1 = torch.distributions.bernoulli.Bernoulli(torch.tensor([inception_prob]))
        if self.training:
            branch1 = self.branch1(x)
            branch2 = self.branch2(x)
            branch3 = self.branch3(x)
            branch4 = self.branch4(x)
            if torch.equal(self.bernouli1.sample(), torch.ones(1)) != 1:
                branch1 = torch.zeros_like(branch1)
            if torch.equal(self.bernouli1.sample(), torch.ones(1)) != 1:
                branch2 = torch.zeros_like(branch2)
            if torch.equal(self.bernouli1.sample(), torch.ones(1)) != 1:
                branch3 = torch.zeros_like(branch3)
            if torch.equal(self.bernouli1.sample(), torch.ones(1)) != 1:
                branch4 = torch.zeros_like(branch4)
                x = [branch1, branch2, branch3, branch4]

            output = torch.cat(x, 1)
            return output
        else:
            branch1 = self.branch1(x)
            branch2 = self.branch2(x)
            branch3 = self.branch3(x)
            branch4 = self.branch4(x)
            x = [branch1, branch2, branch3, branch4]
            output = torch.cat(x, 1)
            return output
            
cfgs = {"V1": ["M", 128, "M", 256, "M", 256, "M"]} # Convolution layers kernel size

def vgg_structure(cfg: list):
    layer = []
    layer.append(nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1))
    layer.append(nn.PReLU())  # 改动1
    in_channel = 64
    for i in cfg:
        if i == "M":
            layer.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            con_2d = nn.Conv2d(in_channel, out_channels=i, kernel_size=3, stride=1, padding=1)
            layer.append(con_2d)
            layer += [nn.PReLU()]
            in_channel = i
    return nn.Sequential(*layer)


class Wediv(nn.Module):
    def __init__(self, size):
        super(Wediv, self).__init__()
        self.berbouli = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.5]))
        self.mask = torch.ones(1, size)
    def forward(self, x):
        if self.training:
            self.indentity = x.detach().cpu().squeeze().numpy().T
            Y_CL, K_optimal, W_optimal, rch = WeDIV2.WeDIV2(self.indentity, w_step=0.2, KList=[4]) # Step and Klist
            group_indices = {}
            self.mask = self.mask.to(device)
            for group_label in range(K_optimal):
                group_indices[group_label] = [i for i, label in enumerate(Y_CL) if label == group_label]
            for group in group_indices:
                if torch.equal(self.berbouli.sample(), torch.ones(1)) != 1:
                    self.mask[0, group_indices[group]] = 0
            result = (x * self.mask.unsqueeze(2).unsqueeze(3))
            self.mask = torch.ones(1, 256)
            return result
        else:
            return x

if __name__ == '__main__':
    net = GDnet("V1")
