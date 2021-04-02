import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


class CLTLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_prec=10, relu_act=True, elu_act=False):
        super(CLTLinear, self).__init__()
        self.n_in = in_features
        self.n_out = out_features

        self.prior_prec = prior_prec

        assert not (relu_act and elu_act) # A single layer can only do either relu or elu activation
        self.relu_act = relu_act
        self.elu_act = elu_act 

        self.bias = nn.Parameter(th.Tensor(out_features))
        self.mu_w = Parameter(th.Tensor(out_features, in_features))
        self.logsig2_w = nn.Parameter(th.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        # TODO: Adapt to the newest pytorch initializations
        stdv = 1. / math.sqrt(self.mu_w.size(1))
        self.mu_w.data.normal_(0, stdv)
        self.logsig2_w.data.zero_().normal_(-9, 0.001)
        self.bias.data.zero_()


    def KL(self, loguniform=False):
        if loguniform:
            k1 = 0.63576; k2 = 1.87320; k3 = 1.48695
            log_alpha = self.logsig2_w - 2 * th.log(self.mu_w.abs() + 1e-8)
            kl = -th.sum(k1 * F.sigmoid(k2 + k3 * log_alpha) - 0.5 * F.softplus(-log_alpha) - k1)
        else:
            logsig2_w = self.logsig2_w.clamp(-11, 11)
            kl = 0.5 * (self.prior_prec * (self.mu_w.pow(2) + logsig2_w.exp()) - logsig2_w - 1 - np.log(self.prior_prec)).sum()
        return kl

    def cdf(self, x, mu=0., sig=1.):
        return 0.5 * (1 + th.erf((x - mu) / (sig * math.sqrt(2))))

    def pdf(self, x, mu=0., sig=1.):
        return (1 / (math.sqrt(2 * math.pi) * sig)) * th.exp(-0.5 * ((x - mu) / sig).pow(2))

    def relu_moments(self, mu, sig):
        alpha = mu / sig
        cdf = self.cdf(alpha)
        pdf = self.pdf(alpha)
        relu_mean = mu * cdf + sig * pdf
        relu_var = (sig.pow(2) + mu.pow(2)) * cdf + mu * sig * pdf - relu_mean.pow(2)
        relu_var.clamp_(1e-8) # Avoid negative variance due to numerics
        return relu_mean, relu_var

    def elu_moments_orig(self, mu, sig):
        # the original method without simplifications
        sig2 = sig.pow(2)
        elu_mean = th.exp(mu.clamp_max(10) + sig2/2) * self.cdf(-(mu + sig2)/sig) - self.cdf(-mu/sig) 
        elu_mean += mu * self.cdf(mu/sig) + sig * self.pdf(mu/sig)
        elu_var = th.exp(2*mu.clamp_max(10) + 2*sig2) * self.cdf(-(mu + 2*sig2)/sig) 
        elu_var += - 2 * th.exp(mu.clamp_max(10) + sig2/2) * self.cdf(-(mu + sig2)/sig) 
        elu_var += self.cdf(-mu/sig)
        elu_var += (sig2 + mu.pow(2)) * self.cdf(mu/sig) + mu * sig * self.pdf(mu/sig)
        elu_var += - elu_mean.pow(2)
        elu_var.clamp_min_(1e-8) # Avoid negative variance due to numerics
        return elu_mean, elu_var

    def elu_moments(self, mu, sig):
        # NOTE: For now it is without alpha or the selu extension!
        # Note: Takes roughly 3x as much time as the relu
        # Clamp the mus to avoid problems in the expectation
        sig2 = sig.pow(2)
        alpha = mu/sig 

        cdf_alpha = self.cdf(alpha)
        pdf_alpha = self.pdf(alpha)
        cdf_malpha = 1 - cdf_alpha
        cdf_malphamsig = self.cdf(-alpha - sig)

        elu_mean = th.exp(mu.clamp_max(10) + sig2/2) * cdf_malphamsig - cdf_malpha
        elu_mean += mu * cdf_alpha + sig * pdf_alpha

        elu_var = th.exp(2*mu.clamp_max(10) + 2*sig2) * self.cdf(-alpha - 2*sig) 
        elu_var += - 2 * th.exp(mu.clamp_max(10) + sig2/2) * cdf_malphamsig 
        elu_var += cdf_malpha
        elu_var += (sig2 + mu.pow(2)) * cdf_alpha + mu * sig * pdf_alpha
        elu_var += - elu_mean.pow(2)
        elu_var.clamp_min_(1e-8) # Avoid negative variance due to numerics
        return elu_mean, elu_var

    def forward(self, mu_inp, var_inp=None):
        s2_w = self.logsig2_w.exp()

        mu_out = F.linear(mu_inp, self.mu_w, self.bias)
        if var_inp is None:
            var_out = F.linear(mu_inp.pow(2), s2_w) + 1e-8
        else:
            var_out = F.linear(var_inp + mu_inp.pow(2), s2_w) + F.linear(var_inp, self.mu_w.pow(2)) + 1e-8

        if self.relu_act:
            mu_out, var_out = self.relu_moments(mu_out, var_out.sqrt())

        if self.elu_act:
            mu_out, var_out = self.elu_moments(mu_out, var_out.sqrt())

        return mu_out, var_out # + 1e-8 Already provided in the moment computation

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.n_in) + ' -> ' \
               + str(self.n_out) \
               + f", activation={self.relu_act or self.elu_act}" \
               + f" ({'relu' if self.relu_act else ('elu' if self.elu_act else '')}))" 


class CLTConv(CLTLinear):
    def __init__(self, in_channels, out_channels, kernel_size, prior_prec=10, stride=1,
                 padding=0, dilation=1, groups=1, relu_act=True, elu_act=False, fixed_prior=False):
        super(CLTLinear, self).__init__()
        self.n_in = in_channels
        self.n_out = out_channels

        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.prior_prec = prior_prec

        assert not (relu_act and elu_act)
        self.relu_act = relu_act
        self.elu_act = elu_act

        self.bias = nn.Parameter(th.Tensor(out_channels))
        self.mu_w = nn.Parameter(th.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.logsig2_w = nn.Parameter(th.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.reset_parameters()


    def reset_parameters(self):
        # TODO: Adapt to the newest pytorch initializations
        n = self.n_in
        for k in range(1, self.kernel_size):
            n *= k
        self.mu_w.data.normal_(0, 1. / math.sqrt(n))
        self.logsig2_w.data.zero_().normal_(-9, 0.001)
        self.bias.data.zero_()


    def forward(self, mu_inp, var_inp=None):
        s2_w = self.logsig2_w.exp()

        mu_out = F.conv2d(mu_inp, self.mu_w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if var_inp is None:
            var_out = F.conv2d(mu_inp.pow(2), s2_w, None, self.stride, self.padding, self.dilation, self.groups) + 1e-8
        else:
            var_out = F.conv2d(var_inp + mu_inp.pow(2), s2_w, None, self.stride, self.padding, self.dilation, self.groups)
            var_out += F.conv2d(var_inp, self.mu_w.pow(2), None, self.stride, self.padding, self.dilation, self.groups) + 1e-8

        if self.relu_act:
            mu_out, var_out = self.relu_moments(mu_out, var_out.sqrt())

        if self.elu_act:
            mu_out, var_out = self.elu_moments(mu_out, var_out.sqrt())

        return mu_out, var_out # + 1e-8 Already provided by elu/relu computation

    def __repr__(self):
        s = ('{name}({n_in}, {n_out}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.elu_act:
            s += ', elu-act=True'
        else:
            s += ', relu-act={relu_act}'
        s += ')'

        return s.format(name=self.__class__.__name__, **self.__dict__)




