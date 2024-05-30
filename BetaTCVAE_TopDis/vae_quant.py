import os
import time
import math
from numbers import Number
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
from torch.autograd import Variable
from torch.utils.data import DataLoader

import lib.dist as dist
import lib.utils as utils
import lib.datasets as dset
from lib.flows import FactorialNormalizingFlow

from elbo_decomposition import elbo_decomposition
# from plot_latent_vs_true import plot_vs_gt_shapes, plot_vs_gt_faces, plot_vs_gt_3dshapes  # noqa: F401

###
import torch.nn.init as init
import numpy as np

from rtd_regularizer import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

init_seed = 23
print (init_seed)
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
np.random.seed(init_seed)
###

class MLPEncoder(nn.Module):
    def __init__(self, output_dim):
        super(MLPEncoder, self).__init__()
        self.output_dim = output_dim

        self.fc1 = nn.Linear(4096, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, output_dim)

        self.conv_z = nn.Conv2d(64, output_dim, 4, 1, 0)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 64 * 64)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = self.fc3(h)
        z = h.view(x.size(0), self.output_dim)
        return z


class MLPDecoder(nn.Module):
    def __init__(self, input_dim):
        super(MLPDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 4096)
        )

    def forward(self, z):
        h = z.view(z.size(0), -1)
        h = self.net(h)
        mu_img = h.view(z.size(0), 1, 64, 64)
        return mu_img


class ConvEncoder(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)  # 32 x 32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)  # 16 x 16
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)  # 4 x 4
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 512, 4)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 1, 64, 64)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        z = self.conv_z(h).view(x.size(0), self.output_dim)
        return z


class ConvDecoder(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, 1, 1, 0)  # 1 x 1
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 64, 4, 1, 0)  # 4 x 4
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 16 x 16
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, 4, 2, 1)  # 32 x 32
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, 1, 4, 2, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        mu_img = self.conv_final(h)
        return mu_img
    
###

class ConvEncoder_dsprites(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder_dsprites, self).__init__()
        self.output_dim = output_dim
#         self.encode = nn.Sequential(
#             nn.Conv2d(1, 32, 4, 2, 1),
#             nn.ReLU(True),
#             nn.Conv2d(32, 32, 4, 2, 1),
#             nn.ReLU(True),
#             nn.Conv2d(32, 64, 4, 2, 1),
#             nn.ReLU(True),
#             nn.Conv2d(64, 64, 4, 2, 1),
#             nn.ReLU(True),
#             nn.Conv2d(64, 128, 4, 1),
#             nn.ReLU(True),
#             nn.Conv2d(128, output_dim, 1)
#         )
        self.encode = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 1),
            nn.ReLU(True),
            nn.Conv2d(512, output_dim, 1)
            ) 
                
    def forward(self, x):
        z = self.encode(x).view(x.size(0), self.output_dim)
        return z 

class ConvDecoder_dsprites(nn.Module):
    def __init__(self, z_dim):
        super(ConvDecoder_dsprites, self).__init__()
        self.z_dim = z_dim
#         self.decode = nn.Sequential(
#             nn.Conv2d(z_dim, 128, 1),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(128, 64, 4),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(64, 64, 4, 2, 1),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(64, 32, 4, 2, 1),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(32, 32, 4, 2, 1),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(32, 1, 4, 2, 1),
#         )
        self.decode = nn.Sequential(
            nn.Conv2d(z_dim, 512, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            )
                
    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        mu_img = self.decode(h)
        return mu_img


class ConvEncoder_3dshapes(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder_3dshapes, self).__init__()
        self.output_dim = output_dim
        self.encode = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),
            nn.ReLU(True),
            nn.Conv2d(256, output_dim, 1)
        )
                
    def forward(self, x):
        z = self.encode(x).view(x.size(0), self.output_dim)
        return z 

class ConvDecoder_3dshapes(nn.Module):
    def __init__(self, z_dim):
        super(ConvDecoder_3dshapes, self).__init__()
        self.z_dim = z_dim
        self.decode = nn.Sequential(
            nn.Conv2d(z_dim, 256, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )
                
    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        mu_img = self.decode(h)
        return mu_img
    
def weight_init(model, mode='normal'):
    if mode == 'kaiming':
        initializer = kaiming_init
    elif mode == 'normal':
        initializer = normal_init
        
    for block in model._modules:
        for m in model._modules[block]:
            initializer(m)
    
def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
###


class VAE(nn.Module):
    def __init__(self, z_dim, use_cuda=False, prior_dist=dist.Normal(), q_dist=dist.Normal(),
                 include_mutinfo=True, tcvae=False, conv=False, conv_dsprites=False, conv_3dshapes=False,
                 conv_mpi3d_complex=False, mss=False):
        super(VAE, self).__init__()

        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.include_mutinfo = include_mutinfo
        self.tcvae = tcvae
        self.lamb = 0
        self.beta = 1
        self.mss = mss
        self.x_dist = dist.Bernoulli()

        # Model-specific
        # distribution family of p(z)
        self.prior_dist = prior_dist
        self.q_dist = q_dist
        # hyperparameters for prior p(z)
        self.register_buffer('prior_params', torch.zeros(self.z_dim, 2))

        # create the encoder and decoder networks
#         if conv:
#             self.encoder = ConvEncoder(z_dim * self.q_dist.nparams)
#             self.decoder = ConvDecoder(z_dim)
#         else:
#             self.encoder = MLPEncoder(z_dim * self.q_dist.nparams)
#             self.decoder = MLPDecoder(z_dim)
        if conv_3dshapes or conv_mpi3d_complex:
            self.encoder = ConvEncoder_3dshapes(z_dim * self.q_dist.nparams)
            self.decoder = ConvDecoder_3dshapes(z_dim)
            weight_init(self.encoder)
            weight_init(self.decoder)
            self.nc = 3
        elif conv_dsprites:
            print ('dsprites')
            self.encoder = ConvEncoder_dsprites(z_dim * self.q_dist.nparams)
            self.decoder = ConvDecoder_dsprites(z_dim)
            weight_init(self.encoder)
            weight_init(self.decoder)
            self.nc = 1
        elif conv:
            self.encoder = ConvEncoder(z_dim * self.q_dist.nparams)
            self.decoder = ConvDecoder(z_dim)
            self.nc = 1
        else:
            self.encoder = MLPEncoder(z_dim * self.q_dist.nparams)
            self.decoder = MLPDecoder(z_dim)
            self.nc = 1

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

    # return prior parameters wrapped in a suitable Variable
    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params

    # samples from the model p(x|z)p(z)
    def model_sample(self, batch_size=1):
        # sample from prior (value will be sampled by guide when computing the ELBO)
        prior_params = self._get_prior_params(batch_size)
        zs = self.prior_dist.sample(params=prior_params)
        # decode the latent code z
        x_params = self.decoder.forward(zs)
        return x_params

    # define the guide (i.e. variational distribution) q(z|x)
    def encode(self, x):
#         x = x.view(x.size(0), 1, 64, 64)
        x = x.view(x.size(0), self.nc, 64, 64)
        # use the encoder to get the parameters used to define q(z|x)
        z_params = self.encoder.forward(x).view(x.size(0), self.z_dim, self.q_dist.nparams)
        # sample the latent code z
        zs = self.q_dist.sample(params=z_params)
        return zs, z_params

    def decode(self, z):
#         x_params = self.decoder.forward(z).view(z.size(0), 1, 64, 64)
        x_params = self.decoder.forward(z).view(z.size(0), self.nc, 64, 64)
        xs = self.x_dist.sample(params=x_params)
        return xs, x_params

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        zs, z_params = self.encode(x)
        xs, x_params = self.decode(zs)
        return xs, x_params, zs, z_params
    
    def forward(self, x):
        xs, x_params, zs, z_params = self.reconstruct_img(x)
        return xs, z_params[:,:,0], z_params[:,:,1]
    
    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std*eps

    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[::M+1] = 1 / N
        W.view(-1)[1::M+1] = strat_weight
        W[M-1, 0] = strat_weight
        return W.log()

    def elbo(self, x, dataset_size):
        # log p(x|z) + log p(z) - log q(z|x)
        batch_size = x.size(0)
#         x = x.view(batch_size, 1, 64, 64)
        x = x.view(batch_size, self.nc, 64, 64)
        prior_params = self._get_prior_params(batch_size)
        x_recon, x_params, zs, z_params = self.reconstruct_img(x)
        logpx = self.x_dist.log_density(x, params=x_params).view(batch_size, -1).sum(1)
        logpz = self.prior_dist.log_density(zs, params=prior_params).view(batch_size, -1).sum(1)
        logqz_condx = self.q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)

        elbo = logpx + logpz - logqz_condx

        if self.beta == 1 and self.include_mutinfo and self.lamb == 0:
            return elbo, elbo.detach()

        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        _logqz = self.q_dist.log_density(
            zs.view(batch_size, 1, self.z_dim),
            z_params.view(1, batch_size, self.z_dim, self.q_dist.nparams)
        )

        if not self.mss:
            # minibatch weighted sampling
            logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
            logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
        else:
            # minibatch stratified sampling
            logiw_matrix = Variable(self._log_importance_weight_matrix(batch_size, dataset_size).type_as(_logqz.data))
            logqz = logsumexp(logiw_matrix + _logqz.sum(2), dim=1, keepdim=False)
            logqz_prodmarginals = logsumexp(
                logiw_matrix.view(batch_size, batch_size, 1) + _logqz, dim=1, keepdim=False).sum(1)

        if not self.tcvae:
            if self.include_mutinfo:
                modified_elbo = logpx - self.beta * (
                    (logqz_condx - logpz) -
                    self.lamb * (logqz_prodmarginals - logpz)
                )
            else:
                modified_elbo = logpx - self.beta * (
                    (logqz - logqz_prodmarginals) +
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)
                )
        else:
            if self.include_mutinfo:
                modified_elbo = logpx - \
                    (logqz_condx - logqz) - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)
            else:
                modified_elbo = logpx - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)

        return modified_elbo, elbo.detach()


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


# for loading and batching datasets
def setup_data_loaders(args, use_cuda=False):
#     if args.dataset == 'shapes':
#         train_set = dset.Shapes()
#     elif args.dataset == 'faces':
#         train_set = dset.Faces()
#     else:
#         raise ValueError('Unknown dataset ' + str(args.dataset))
    if args.dataset == 'shapes':
        train_set = dset.Shapes()
    elif args.dataset == 'faces':
        train_set = dset.Faces()
    elif args.dataset == '3dshapes':
        train_set = dset.Shapes3D()
    elif args.dataset == 'mpi3d_complex':
        train_set = dset.MPI3D_complex()
    else:
        raise ValueError('Unknown dataset ' + str(args.dataset))

    kwargs = {'num_workers': 4, 'pin_memory': use_cuda}
    train_loader = DataLoader(dataset=train_set,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    return train_loader


win_samples = None
win_test_reco = None
win_latent_walk = None
win_train_elbo = None


def display_samples(model, x, vis):
    global win_samples, win_test_reco, win_latent_walk

    # plot random samples
    sample_mu = model.model_sample(batch_size=100).sigmoid()
    sample_mu = sample_mu
#     images = list(sample_mu.view(-1, 1, 64, 64).data.cpu())
    images = sample_mu.view(-1, model.nc, 64, 64).data.cpu()
    win_samples = vis.images(images, 10, 2, opts={'caption': 'samples'}, win=win_samples)

    # plot the reconstructed distribution for the first 50 test images
    test_imgs = x[:50, :]
    _, reco_imgs, zs, _ = model.reconstruct_img(test_imgs)
    reco_imgs = reco_imgs.sigmoid()
#     test_reco_imgs = torch.cat([
#         test_imgs.view(1, -1, 64, 64), reco_imgs.view(1, -1, 64, 64)], 0).transpose(0, 1)
    test_reco_imgs = torch.cat([
        test_imgs.view(-1, model.nc, 64, 64), reco_imgs.view(-1, model.nc, 64, 64)], 0)
#     win_test_reco = vis.images(
#         list(test_reco_imgs.contiguous().view(-1, 1, 64, 64).data.cpu()), 10, 2,
#         opts={'caption': 'test reconstruction image'}, win=win_test_reco)
    win_test_reco = vis.images(
        test_reco_imgs.contiguous().view(-1, model.nc, 64, 64).data.cpu(), 10, 2,
        opts={'caption': 'test reconstruction image'}, win=win_test_reco)

    # plot latent walks (change one variable while all others stay the same)
    zs = zs[0:3]
    batch_size, z_dim = zs.size()
    xs = []
#     delta = torch.autograd.Variable(torch.linspace(-2, 2, 7), volatile=True).type_as(zs)
    delta = torch.linspace(-2, 2, 7).type_as(zs)
    for i in range(z_dim):
#         vec = Variable(torch.zeros(z_dim)).view(1, z_dim).expand(7, z_dim).contiguous().type_as(zs)
        vec = torch.zeros(z_dim).view(1, z_dim).expand(7, z_dim).contiguous().type_as(zs)
        vec[:, i] = 1
        vec = vec * delta[:, None]
        zs_delta = zs.clone().view(batch_size, 1, z_dim)
        zs_delta[:, :, i] = 0
        zs_walk = zs_delta + vec[None]
        xs_walk = model.decoder.forward(zs_walk.view(-1, z_dim)).sigmoid()
        xs.append(xs_walk)

#     xs = list(torch.cat(xs, 0).data.cpu())
    xs = torch.cat(xs, 0).data.cpu()
    win_latent_walk = vis.images(xs, 7, 2, opts={'caption': 'latent walk'}, win=win_latent_walk)


def plot_elbo(train_elbo, vis):
    global win_train_elbo
    win_train_elbo = vis.line(torch.Tensor(train_elbo), opts={'markers': True}, win=win_train_elbo)


def anneal_kl(args, vae, iteration):
    if args.dataset == 'shapes':
        warmup_iter = 7000
    elif args.dataset == 'faces':
        warmup_iter = 2500
    ###
    elif args.dataset == '3dshapes':
        warmup_iter = 4000
    elif args.dataset == 'mpi3d_complex':
        warmup_iter = 4000
    ###

    if args.lambda_anneal:
        vae.lamb = max(0, 0.95 - 1 / warmup_iter * iteration)  # 1 --> 0
    else:
        vae.lamb = 0
    if args.beta_anneal:
        vae.beta = min(args.beta, args.beta / warmup_iter * iteration)  # 0 --> 1
    else:
        vae.beta = args.beta


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
#     parser.add_argument('-d', '--dataset', default='shapes', type=str, help='dataset name',
#         choices=['shapes', 'faces'])
    parser.add_argument('-d', '--dataset', default='shapes', type=str, help='dataset name',
        choices=['shapes', 'faces', '3dshapes', 'mpi3d_complex'])
    parser.add_argument('-dist', default='normal', type=str, choices=['normal', 'laplace', 'flow'])
    parser.add_argument('-n', '--num-epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=2048, type=int, help='batch size')
    parser.add_argument('-l', '--learning-rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-z', '--latent-dim', default=10, type=int, help='size of latent dimension')
    parser.add_argument('--beta', default=1, type=float, help='ELBO penalty term')
    parser.add_argument('--tcvae', action='store_true')
    parser.add_argument('--exclude-mutinfo', action='store_true')
    parser.add_argument('--beta-anneal', action='store_true')
    parser.add_argument('--lambda-anneal', action='store_true')
    parser.add_argument('--mss', action='store_true', help='use the improved minibatch estimator')
    parser.add_argument('--conv', action='store_true')
    ###
    parser.add_argument('--conv_dsprites', action='store_true')
    parser.add_argument('--conv_3dshapes', action='store_true')
    parser.add_argument('--conv_mpi3d_complex', action='store_true')
    ###
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--visdom', action='store_true', help='whether plotting in visdom is desired')
    parser.add_argument('--save', default='test1')
    parser.add_argument('--log_freq', default=1000, type=int, help='num iterations per log')
    
    ### RTD specific arguments
    parser.add_argument('--use_rtd', action='store_true', help='use rtd regularization')
    parser.add_argument('--lp', default=1, type=int, help='norm to compute topological features')
    parser.add_argument('--q_normalize', action='store_true', help='whether to use quantile normalization')
    parser.add_argument('--gamma_rtd', default=1, type=float, help='gamma hyperparameter for rtd regularization')
    parser.add_argument('--sample_based', action='store_true', help='compute sample based rtd regularization')
    parser.add_argument('--weightnorm_sampler', action='store_true', help='whether to sample dim according to weight norm')
    parser.add_argument('--delay_iter', default=0, type=float, help='number of iterations to wait before optimize topdis loss')
    ###
    args = parser.parse_args()
    
    print (args)

    torch.cuda.set_device(args.gpu)

    # data loader
    train_loader = setup_data_loaders(args, use_cuda=True)

    # setup the VAE
    if args.dist == 'normal':
        prior_dist = dist.Normal()
        q_dist = dist.Normal()
    elif args.dist == 'laplace':
        prior_dist = dist.Laplace()
        q_dist = dist.Laplace()
    elif args.dist == 'flow':
        prior_dist = FactorialNormalizingFlow(dim=args.latent_dim, nsteps=32)
        q_dist = dist.Normal()

#     vae = VAE(z_dim=args.latent_dim, use_cuda=True, prior_dist=prior_dist, q_dist=q_dist,
#         include_mutinfo=not args.exclude_mutinfo, tcvae=args.tcvae, conv=args.conv, mss=args.mss)
    vae = VAE(z_dim=args.latent_dim, use_cuda=True, prior_dist=prior_dist, q_dist=q_dist,
        include_mutinfo=not args.exclude_mutinfo, tcvae=args.tcvae, conv=args.conv,
        conv_dsprites=args.conv_dsprites, conv_3dshapes=args.conv_3dshapes, conv_mpi3d_complex=args.conv_mpi3d_complex,
        mss=args.mss)
    
    print(vae.encoder.encode[0].weight[0,0].data)
    print(vae.decoder.decode[0].weight[0,0].data)
    
    # RTD regularization
    use_rtd = args.use_rtd
    lp = args.lp
    q_normalize = args.q_normalize
    gamma_rtd = args.gamma_rtd
    sample_based = args.sample_based
    weightnorm_sampler = args.weightnorm_sampler
    rtd_reg = None
    rtd_iter_start = args.delay_iter

    if use_rtd:
        rtd_reg = RTDRegularizer(lp, q_normalize, sample_based, weightnorm_sampler)
        print ('Use RTD regularization with lp = {}, q_normalize = {}, gamma_rtd = {}, sample_based = {}, weightnorm_sampler = {}'.format(lp, q_normalize, gamma_rtd, sample_based, weightnorm_sampler))

    # setup the optimizer
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)

    # setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom(env=args.save, port=8097)

    train_elbo = []

    # training loop
    dataset_size = len(train_loader.dataset)
    num_iterations = len(train_loader) * args.num_epochs
    iteration = 0
    # initialize loss accumulator
    elbo_running_mean = utils.RunningAverageMeter()
    while iteration < num_iterations:
        for i, x in enumerate(train_loader):
            iteration += 1
            batch_time = time.time()
            vae.train()
            anneal_kl(args, vae, iteration)
            optimizer.zero_grad()
            # transfer to GPU
#             x = x.cuda(async=True)
            x = x.cuda(non_blocking=True)
            # wrap the mini-batch in a PyTorch Variable
            x = Variable(x)
            # do ELBO gradient and accumulate loss
            obj, elbo = vae.elbo(x, dataset_size)
            if utils.isnan(obj).any():
                raise ValueError('NaN spotted in objective.')
            
            if use_rtd and rtd_reg is not None:
                if iteration >= rtd_iter_start:
                    i, rtd_loss = rtd_reg.compute_reg(vae, x, None)
                    topdis = gamma_rtd * rtd_loss
            else:
                topdis = 0.
                
            (obj.mean().mul(-1) + topdis).backward()
#             elbo_running_mean.update(elbo.mean().data[0])
            elbo_running_mean.update(elbo.mean().item())
            optimizer.step()

            # report training diagnostics
            if iteration % args.log_freq == 0:
                train_elbo.append(elbo_running_mean.avg)
                print('[iteration %03d] time: %.2f \tbeta %.2f \tlambda %.2f training ELBO: %.4f (%.4f)\tTopDis %.4f' % (
                    iteration, time.time() - batch_time, vae.beta, vae.lamb,
                    elbo_running_mean.val, elbo_running_mean.avg, topdis.item()))

                vae.eval()

                # plot training and test ELBOs
                if args.visdom:
                    display_samples(vae, x, vis)
                    plot_elbo(train_elbo, vis)

                if iteration % (50 * args.log_freq) == 0:
                    utils.save_checkpoint({
                        'state_dict': vae.state_dict(),
                        'args': args}, args.save, iteration)
#                 eval('plot_vs_gt_' + args.dataset)(vae, train_loader.dataset,
#                     os.path.join(args.save, 'gt_vs_latent_{:05d}.png'.format(iteration)))

    # Report statistics after training
    vae.eval()
    utils.save_checkpoint({
        'state_dict': vae.state_dict(),
        'args': args}, args.save, iteration)
    dataset_loader = DataLoader(train_loader.dataset, batch_size=1000, num_workers=1, shuffle=False)
    logpx, dependence, information, dimwise_kl, analytical_cond_kl, marginal_entropies, joint_entropy = \
        elbo_decomposition(vae, dataset_loader)
    torch.save({
        'logpx': logpx,
        'dependence': dependence,
        'information': information,
        'dimwise_kl': dimwise_kl,
        'analytical_cond_kl': analytical_cond_kl,
        'marginal_entropies': marginal_entropies,
        'joint_entropy': joint_entropy
    }, os.path.join(args.save, 'elbo_decomposition.pth'))
#     eval('plot_vs_gt_' + args.dataset)(vae, dataset_loader.dataset, os.path.join(args.save, 'gt_vs_latent.png'))
    return vae


if __name__ == '__main__':
    model = main()
