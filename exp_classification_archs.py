import math
import numpy as np
import torch as th
import torch.nn.functional as F
from CLTLayer import CLTLinear, CLTConv
from VBLayer import VBLinear, VBConv
from torch import nn
from utils import Phi
from tqdm import tqdm

class BEDL(nn.Module):
    def __init__(self, n_channels, n_classes, delta=0.1, prior_prec=10):
        super(BEDL, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.prior_prec = prior_prec
        self.n_samples = 20

        relu_act = True 
        elu_act = False

        if n_channels == 1:
            self.conv1 = CLTConv(1, 20, 5, stride=2, prior_prec=prior_prec, relu_act=relu_act, elu_act=elu_act)
            self.conv2 = CLTConv(20, 50, 5, stride=2, prior_prec=prior_prec, relu_act=relu_act, elu_act=elu_act)
            self.dense1 = CLTLinear(4*4*50, 500, prior_prec=prior_prec, relu_act=relu_act, elu_act=elu_act)
            self.dense2 = CLTLinear(500, n_classes, prior_prec=prior_prec, relu_act=False, elu_act=elu_act)

        elif n_channels == 3:
            self.conv1 = CLTConv(3, 192, 5, stride=2, prior_prec=prior_prec, relu_act=relu_act, elu_act=elu_act)
            self.conv2 = CLTConv(192, 192, 5, stride=2, prior_prec=prior_prec, relu_act=relu_act, elu_act=elu_act)
            self.dense1 = CLTLinear(5 * 5 * 192, 1000, prior_prec=prior_prec, relu_act=relu_act, elu_act=elu_act)
            self.dense2 = CLTLinear(1000, n_classes, prior_prec=prior_prec, relu_act=False, elu_act=False)

    def forward(self, inp):
        mu_h, var_h = self.conv1(inp, None)
        mu_h, var_h = self.conv2(mu_h, var_h)
        if self.n_channels == 1:
            mu_h, var_h = mu_h.view(-1, 4*4*50), var_h.view(-1, 4*4*50)
        else:
            mu_h, var_h = mu_h.view(-1, 5*5*192), var_h.view(-1, 5*5*192)
        mu_h, var_h = self.dense1(mu_h, var_h)
        mu_h, var_h = self.dense2(mu_h, var_h)
        return mu_h, var_h

    def predict(self, data):
        mu, var = self.forward(data)
        
        prob = sum([F.softmax(mu + var.sqrt() * th.randn_like(mu), 1) for _ in range(self.n_samples)]) / self.n_samples
        return prob

    def loss(self, data, target, n_train):

        if target.is_cuda:
            target = th.eye(self.n_classes).cuda()[target]
        else:
            target = th.eye(self.n_classes)[target]

        prob = self.predict(data)
        log_probs = th.log(prob + 1e-8)


        data_fit = - th.sum(target * log_probs, 1).mean()
        
        return data_fit 


class BEDLPAC(nn.Module):
    def __init__(self, n_channels, n_classes, delta=0.1, prior_prec=10):
        super(BEDLPAC, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.delta = delta 
        self.prior_prec = prior_prec
        self.n_samples = 5

        relu_act = True 
        elu_act = False

        if n_channels == 1:
            self.conv1 = CLTConv(1, 20, 5, stride=2, prior_prec=prior_prec, relu_act=relu_act, elu_act=elu_act)
            self.conv2 = CLTConv(20, 50, 5, stride=2, prior_prec=prior_prec, relu_act=relu_act, elu_act=elu_act)
            self.dense1 = CLTLinear(4*4*50, 500, prior_prec=prior_prec, relu_act=relu_act, elu_act=elu_act)
            self.dense2 = CLTLinear(500, n_classes, prior_prec=prior_prec, relu_act=False, elu_act=elu_act)

        elif n_channels == 3:
            self.conv1 = CLTConv(3, 192, 5, stride=2, prior_prec=prior_prec, relu_act=relu_act, elu_act=elu_act)
            self.conv2 = CLTConv(192, 192, 5, stride=2, prior_prec=prior_prec, relu_act=relu_act, elu_act=elu_act)
            self.dense1 = CLTLinear(5 * 5 * 192, 1000, prior_prec=prior_prec, relu_act=relu_act, elu_act=elu_act)
            self.dense2 = CLTLinear(1000, n_classes, prior_prec=prior_prec, relu_act=False, elu_act=False)

    def forward(self, inp):
        mu_h, var_h = self.conv1(inp, None)
        mu_h, var_h = self.conv2(mu_h, var_h)
        if self.n_channels == 1:
            mu_h, var_h = mu_h.view(-1, 4*4*50), var_h.view(-1, 4*4*50)
        else:
            mu_h, var_h = mu_h.view(-1, 5*5*192), var_h.view(-1, 5*5*192)
        mu_h, var_h = self.dense1(mu_h, var_h)
        mu_h, var_h = self.dense2(mu_h, var_h)
        return mu_h, var_h

    def predict(self, data):
        mu, var = self.forward(data)
        
        prob = sum([F.softmax(mu + var.sqrt() * th.randn_like(mu), 1) for _ in range(self.n_samples)]) / self.n_samples
        return prob

    def loss(self, data, target, n_train):
        mu_pred, var_pred = self.forward(data)

        kl = 0
        for _ in range(self.n_samples):
            alpha = th.exp(mu_pred + var_pred.sqrt() * th.randn_like(mu_pred))
            alpha_0 = alpha.sum(1) # I.e. the number of classes
            tmp = th.lgamma(alpha_0) - th.lgamma(alpha).sum(1) - math.lgamma(alpha.shape[1])
            tmp += ((alpha - 1) * (th.digamma(alpha) - th.digamma(alpha_0[:,None]))).sum(1)
            kl += tmp
        kl /= self.n_samples 
        kl = kl.mean() 

        if target.is_cuda:
            target = th.eye(self.n_classes).cuda()[target]
        else:
            target = th.eye(self.n_classes)[target]

        prob = sum([F.softmax(mu_pred + var_pred.sqrt() * th.randn_like(mu_pred), 1) for _ in range(self.n_samples)]) / self.n_samples
        log_probs = th.log(prob + 1e-8)
        data_fit = - th.sum(target * log_probs, 1).mean()
        B = n_train
        reg_term = th.sqrt((kl + math.log(B) - math.log(self.delta))/n_train)
        
        return data_fit + reg_term


class VBNet(nn.Module):
    def __init__(self, n_channels, n_classes, loguniform=False, prior_prec=10):
        super(VBNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.loguniform = loguniform
        self.n_samples = 5
        self.prior_prec=prior_prec

        if n_channels == 1:
            self.conv1 = VBConv(1, 20, 5, stride=2, prior_prec=prior_prec)
            self.conv2 = VBConv(20, 50, 5, stride=2, prior_prec=prior_prec)
            self.dense1 = VBLinear(4*4*50, 500, prior_prec=prior_prec)
            self.dense2 = VBLinear(500, n_classes, prior_prec=prior_prec)

        elif n_channels == 3:
            self.conv1 = VBConv(3, 192, 5, stride=2, prior_prec=prior_prec)
            self.conv2 = VBConv(192, 192, 5, stride=2, prior_prec=prior_prec)
            self.dense1 = VBLinear(5 * 5 * 192, 1000, prior_prec=prior_prec)
            self.dense2 = VBLinear(1000, n_classes, prior_prec=prior_prec)


    def forward(self, inp):
        out = F.relu(self.conv1(inp))
        out = F.relu(self.conv2(out))
        if self.n_channels == 1:
            out = out.view(-1, 4*4*50)
        else:
            out = out.view(-1, 5*5*192)
        out = F.relu(self.dense1(out))
        out = self.dense2(out)
        return out

    def predict(self, data):
        self.sample = True
        if self.sample:
            prob = sum([F.softmax(self.forward(data), 1) for _ in range(self.n_samples)]) / self.n_samples
            return prob

    def loss(self, data, target, n_train):

        kl = sum(l.KL(loguniform=self.loguniform) for l in [self.conv1, self.conv2, self.dense1, self.dense2])

        if target.is_cuda:
            target = th.eye(self.n_classes).cuda()[target]
        else:
            target = th.eye(self.n_classes)[target]

        log_probs = sum([F.log_softmax(self.forward(data), 1) for _ in range(self.n_samples)]) / self.n_samples

        data_fit = - th.sum(target * log_probs, 1).mean()

        return data_fit + kl/n_train


class DropNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(DropNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        if n_channels == 1:
            self.conv1 = nn.Conv2d(1, 20, 5, stride=2)
            self.conv2 = nn.Conv2d(20, 50, 5, stride=2)
            self.dense1 = nn.Linear(4*4*50, 500)
            self.dense2 = nn.Linear(500, n_classes)

        elif n_channels == 3:
            self.conv1 = nn.Conv2d(3, 192, 5, stride=2)
            self.conv2 = nn.Conv2d(192, 192, 5, stride=2)
            self.dense1 = nn.Linear(5 * 5 * 192, 1000)
            self.dense2 = nn.Linear(1000, n_classes)

    def forward(self, inp):
        drop_rate = 0.5
        out = F.dropout(F.relu(self.conv1(inp)), drop_rate, self.training)
        out = F.dropout(F.relu(self.conv2(out)), drop_rate, self.training)
        if self.n_channels == 1:
            out = out.view(-1, 4*4*50)
        else:
            out = out.view(-1, 5*5*192)
        out = F.dropout(F.relu(self.dense1(out)), drop_rate, self.training)
        out = self.dense2(out)
        return out
    
    def predict(self, data):
        self.sample = True
        if self.sample:
            prob = sum([F.softmax(self.forward(data), 1) for _ in range(20)]) / 20
            return prob
    def loss(self, data, target, n_train):

        if target.is_cuda:
            target = th.eye(self.n_classes).cuda()[target]
        else:
            target = th.eye(self.n_classes)[target]

        n_samples = 3
        log_probs = sum([F.log_softmax(self.forward(data), 1) for _ in range(n_samples)]) / n_samples
        data_fit = - th.sum(target * log_probs, 1).mean()

        return data_fit


class DVINet(nn.Module):
    def __init__(self, n_channels, n_classes, prior_prec=10):
        super(DVINet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.prior_prec = prior_prec

        if n_channels == 1:
            self.conv1 = CLTConv(1, 20, 5, stride=2, prior_prec=prior_prec)
            self.conv2 = CLTConv(20, 50, 5, stride=2, prior_prec=prior_prec)
            self.dense1 = CLTLinear(4*4*50, 500, prior_prec=prior_prec)
            self.dense2 = CLTLinear(500, n_classes, relu_act=False, prior_prec=prior_prec)

        elif n_channels == 3:
            self.conv1 = CLTConv(3, 192, 5, stride=2, prior_prec=prior_prec)
            self.conv2 = CLTConv(192, 192, 5, stride=2, prior_prec=prior_prec)
            self.dense1 = CLTLinear(5 * 5 * 192, 1000, prior_prec=prior_prec)
            self.dense2 = CLTLinear(1000, n_classes, relu_act=False, prior_prec=prior_prec)

    def forward(self, inp):
        mu_h, var_h = self.conv1(inp, None)
        mu_h, var_h = self.conv2(mu_h, var_h)
        if self.n_channels == 1:
            mu_h, var_h = mu_h.view(-1, 4*4*50), var_h.view(-1, 4*4*50)
        else:
            mu_h, var_h = mu_h.view(-1, 5*5*192), var_h.view(-1, 5*5*192)
        mu_h, var_h = self.dense1(mu_h, var_h)
        mu_h, var_h = self.dense2(mu_h, var_h)
        return mu_h, var_h

    def predict(self, data):
        mu, var = self.forward(data)
        self.sample = True
        if self.sample:
            prob = sum([F.softmax(mu + var.sqrt() * th.randn_like(mu), 1) for _ in range(20)]) / 20
            return prob

    def loss(self, data, target, n_train):
        mu, var = self.forward(data)

        kl = sum(l.KL() for l in [self.conv1, self.conv2, self.dense1, self.dense2])

        if target.is_cuda:
            target = th.eye(self.n_classes).cuda()[target]
        else:
            target = th.eye(self.n_classes)[target]


        n_samples = 20
        log_probs = sum([F.log_softmax(mu + var.sqrt() * th.randn_like(mu), 1) for _ in range(n_samples)]) / n_samples
        data_fit = - th.sum(target * log_probs, 1).mean()

        return data_fit + kl/n_train
