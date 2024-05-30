# This code is based on the code from https://github.com/besterma/dava

from __future__ import annotations

from torch import Tensor, device, dtype
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List

import torch
from torchtyping import TensorType
import torch.nn as nn
import torch.nn.functional
import numpy as np
import torch.nn.init as init

def compute_gaussian_kl(z_mean: TensorType["batch", "z_dim"], z_logvar: TensorType["batch", "z_dim"]):
    # z_sigma = torch.exp(z_logvar / 2)
    # therefore "0.5 * (z_sigma ** 2 + z_mean ** 2 - 2 * torch.log(z_sigma) - 1).mean()" becomes:
    return 0.5 * (torch.square(z_mean) + torch.exp(z_logvar) - z_logvar - 1)


def gaussian_log_density(z_sampled: TensorType["batch", "num_latents"],
                         z_mean: TensorType["batch", "num_latents"],
                         z_logvar: TensorType["batch", "num_latents"]):
    normalization = torch.log(torch.tensor(2. * np.pi))
    inv_sigma = torch.exp(-z_logvar)
    tmp = (z_sampled - z_mean)
    return -0.5 * (tmp * tmp * inv_sigma + z_logvar + normalization)


def total_correlation(z: TensorType["batch", "num_latents"],
                      z_mean: TensorType["batch", "num_latents"],
                      z_logvar: TensorType["batch", "num_latents"],
                      dataset_size: int) -> Tensor:
    log_qz_prob = gaussian_log_density(z.unsqueeze(1), z_mean.unsqueeze(0), z_logvar.unsqueeze(0))

    log_qz_product = torch.sum(
        torch.logsumexp(log_qz_prob, dim=1),
        dim=1
    )
    log_qz = torch.logsumexp(
        torch.sum(log_qz_prob, dim=2),
        dim=1
    )
    return torch.mean(log_qz - log_qz_product)

class ConvEncoder(nn.Module):
    def __init__(self, output_dim: int, num_channels: int, use_batch_norm: bool = False,
                 use_spectral_norm: bool = False, use_layer_norm: bool = False,
                 use_instance_norm: bool = False, scale_factor=1, arch_type=1, dataset_name='dsprites'):
        super(ConvEncoder, self).__init__()
        self.output_dim = output_dim
        self.num_channels = num_channels
        self.use_batch_norm = use_batch_norm
        self.use_spectral_norm = use_spectral_norm
        self.use_layer_norm = use_layer_norm
        self.use_instance_norm = use_instance_norm
        self.scale_factor = scale_factor
        assert not (use_layer_norm and use_batch_norm), "Cant use both layer and batch norm" 
        
        if dataset_name == 'dsprites':
            # dsprites
            if arch_type == 1:
                self.encode = nn.Sequential(
                    nn.Conv2d(1, 32, 4, 2, 1),
                    nn.ReLU(True),
                    nn.InstanceNorm2d(32) if self.use_instance_norm else nn.Identity(),
                    nn.Conv2d(32, 32, 4, 2, 1),
                    nn.ReLU(True),
                    nn.InstanceNorm2d(32) if self.use_instance_norm else nn.Identity(),
                    nn.Conv2d(32, 64, 4, 2, 1),
                    nn.ReLU(True),
                    nn.InstanceNorm2d(64) if self.use_instance_norm else nn.Identity(),
                    nn.Conv2d(64, 64, 4, 2, 1),
                    nn.ReLU(True),
                    nn.InstanceNorm2d(64) if self.use_instance_norm else nn.Identity(),
                    nn.Conv2d(64, 128, 4, 1),
                    nn.ReLU(True),
                    nn.BatchNorm2d(128) if self.use_instance_norm else nn.Identity(),
                    nn.Conv2d(128, int(2*output_dim), 1)
                )
            elif arch_type == 2:
                self.encode = nn.Sequential(
                    nn.Conv2d(1, 32, 4, 2, 1),
                    nn.ReLU(True),
                    nn.InstanceNorm2d(32) if self.use_instance_norm else nn.Identity(),
                    nn.Conv2d(32, 64, 4, 2, 1),
                    nn.ReLU(True),
                    nn.InstanceNorm2d(64) if self.use_instance_norm else nn.Identity(),
                    nn.Conv2d(64, 128, 4, 2, 1),
                    nn.ReLU(True),
                    nn.InstanceNorm2d(128) if self.use_instance_norm else nn.Identity(),
                    nn.Conv2d(128, 256, 4, 2, 1),
                    nn.ReLU(True),
                    nn.InstanceNorm2d(256) if self.use_instance_norm else nn.Identity(),
                    nn.Conv2d(256, 512, 4, 1),
                    nn.ReLU(True),
                    nn.BatchNorm2d(512) if self.use_instance_norm else nn.Identity(),
                    nn.Conv2d(512, int(2*output_dim), 1)
                )
        elif dataset_name in ['mpi3d', '3dshapes']:
            # 3D Shapes, MPI 3D
            self.encode = nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1),
                nn.ReLU(True),
                nn.InstanceNorm2d(32) if self.use_instance_norm else nn.Identity(),
                nn.Conv2d(32, 32, 4, 2, 1),
                nn.ReLU(True),
                nn.InstanceNorm2d(32) if self.use_instance_norm else nn.Identity(),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.ReLU(True),
                nn.InstanceNorm2d(64) if self.use_instance_norm else nn.Identity(),
                nn.Conv2d(64, 64, 4, 2, 1),
                nn.ReLU(True),
                nn.InstanceNorm2d(64) if self.use_instance_norm else nn.Identity(),
                nn.Conv2d(64, 256, 4, 1),
                nn.ReLU(True),
                nn.BatchNorm2d(256) if self.use_instance_norm else nn.Identity(),
                nn.Conv2d(256, int(2*output_dim), 1)
            )
        elif dataset_name == '3dfaces':
            # 3D Faces
            self.encode = nn.Sequential(
                nn.Conv2d(1, 32, 4, 2, 1),
                nn.ReLU(True),
                nn.InstanceNorm2d(32) if self.use_instance_norm else nn.Identity(),
                nn.Conv2d(32, 32, 4, 2, 1),
                nn.ReLU(True),
                nn.InstanceNorm2d(32) if self.use_instance_norm else nn.Identity(),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.ReLU(True),
                nn.InstanceNorm2d(64) if self.use_instance_norm else nn.Identity(),
                nn.Conv2d(64, 64, 4, 2, 1),
                nn.ReLU(True),
                nn.InstanceNorm2d(64) if self.use_instance_norm else nn.Identity(),
                nn.Conv2d(64, 256, 4, 1),
                nn.ReLU(True),
                nn.BatchNorm2d(256) if self.use_instance_norm else nn.Identity(),
                nn.Conv2d(256, int(2*output_dim), 1)
            )
            
    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, x):
        stats = self.encode(x)
        mu = stats[:, :self.output_dim, 0, 0]
        logvar = stats[:, self.output_dim:, 0, 0]
        return mu, logvar
    
    def state_dict(self, destination=None, prefix: str = '', keep_vars: bool = False):
        state_dict = super(ConvEncoder, self).state_dict(destination, prefix, keep_vars)
        state_dict["use_batch_norm"] = self.use_batch_norm
        state_dict["use_layer_norm"] = self.use_layer_norm
        state_dict["use_instance_norm"] = self.use_instance_norm
        return state_dict


class Discriminator(ConvEncoder):
    def __init__(self, output_dim: int, num_channels: int, use_batch_norm: bool = False,
                 use_spectral_norm: bool = False, use_layer_norm: bool = False,
                 use_instance_norm: bool = False, scale_factor: int = 1, dataset_name='dsprites'):
        super(Discriminator, self).__init__(output_dim, num_channels, use_batch_norm,
                                            use_spectral_norm, use_layer_norm, use_instance_norm,
                                            scale_factor, arch_type=2 if dataset_name=='dsprites' else 1,
                                            dataset_name=dataset_name)
        self.output_act = nn.Sigmoid()

    def forward(self, x: TensorType["batch", "num_channels", "x", "y"]):
        z, _ = super(Discriminator, self).forward(x)
        return z

class ConvDecoder(nn.Module):
    def __init__(self, input_dim: int, num_channels: int, use_batch_norm: bool = False,
                 use_spectral_norm: bool = False, use_layer_norm: bool = False,
                 use_instance_norm: bool = False, scale_factor=1, arch_type=1, dataset_name='dsprites'):
        super(ConvDecoder, self).__init__()
        self.input_dim = input_dim
        self.num_channels = num_channels
        self.use_batch_norm = use_batch_norm  # https://discuss.pytorch.org/t/autocast-with-normalization-layers/94125
        self.use_spectral_norm = use_spectral_norm
        self.use_layer_norm = use_layer_norm
        self.use_instance_norm = use_instance_norm
        self.scale_factor = scale_factor
        assert not (use_layer_norm and use_batch_norm), "Cant use both layer and batch norm"
        
        if dataset_name == 'dsprites':
            # dsprites
            if arch_type == 1:
                self.decode = nn.Sequential(
                    nn.Conv2d(input_dim, 128, 1),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(128, 64, 4),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(64, 64, 4, 2, 1),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(64, 32, 4, 2, 1),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(32, 32, 4, 2, 1),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(32, 1, 4, 2, 1),
                )
            elif arch_type == 2:
                self.decode = nn.Sequential(
                    nn.Conv2d(input_dim, 512, 1),
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
        elif dataset_name in ['mpi3d', '3dshapes']:
            # 3D Shapes, MPI 3D
            self.decode = nn.Sequential(
                nn.Conv2d(input_dim, 256, 1),
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
        elif dataset_name == '3dfaces':
            # 3D Faces
            self.decode = nn.Sequential(
                nn.Conv2d(input_dim, 256, 1),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 64, 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 64, 4, 2, 1),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 32, 4, 2, 1),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 1, 4, 2, 1),
            )
    
    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)
            
    def forward(self, z):
        z = z.view(-1, self.input_dim, 1, 1)
        x_recon = self.decode(z)
        return x_recon
    
    def state_dict(self, destination = None, prefix: str = '', keep_vars: bool = False):
        state_dict = super(ConvDecoder, self).state_dict(destination, prefix, keep_vars)
        state_dict["use_batch_norm"] = self.use_batch_norm
        state_dict["use_layer_norm"] = self.use_layer_norm
        state_dict["use_instance_norm"] = self.use_instance_norm
        return state_dict

class EncoderDecoderModel(nn.Module):
    def __init__(self, z_dim: int, num_channels: int, use_batch_norm: bool = False,
                 use_spectral_norm: bool = False, use_layer_norm: bool = False,
                 use_instance_norm: bool = False, scale_factor=1, dataset_name='dsprites'):
        super(EncoderDecoderModel, self).__init__()
        self.encoder = ConvEncoder(z_dim, num_channels, use_batch_norm,
                                   use_spectral_norm, use_layer_norm, use_instance_norm,
                                   scale_factor, arch_type=1, dataset_name=dataset_name)
        self.decoder = ConvDecoder(z_dim, num_channels, use_batch_norm,
                                   use_spectral_norm, use_layer_norm, use_instance_norm,
                                   scale_factor, arch_type=1, dataset_name=dataset_name)
        self.encoder.weight_init()
        self.decoder.weight_init()
        if dataset_name == 'dsprites':
            self.encoder = ConvEncoder(z_dim, num_channels, use_batch_norm,
                                       use_spectral_norm, use_layer_norm, use_instance_norm,
                                       scale_factor, arch_type=2, dataset_name=dataset_name)
            self.decoder = ConvDecoder(z_dim, num_channels, use_batch_norm,
                                       use_spectral_norm, use_layer_norm, use_instance_norm,
                                       scale_factor, arch_type=2, dataset_name=dataset_name)
            self.encoder.weight_init()
            self.decoder.weight_init()

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


class FVAEDiscriminator(nn.Module):
    def __init__(self, z_dim):
        super(FVAEDiscriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 2),
        )

    def forward(self, z):
        return self.net(z).squeeze()


class BetaVAE(nn.Module):
    def __init__(self, z_dim: int, num_channels: int, beta: int, use_batch_norm: bool = False,
                 use_spectral_norm: bool = False, use_layer_norm: bool = False,
                 use_instance_norm: bool = False, scale_factor: int = 1, dataset_name='dsprites', **kwargs):
        super(BetaVAE, self).__init__(**kwargs)
        assert beta > 0
        assert num_channels > 0
        assert z_dim > 0
        self.z_dim = z_dim
        self.num_channesl = num_channels
        self.beta = beta
        self.enc_dec_model = EncoderDecoderModel(z_dim, num_channels, use_batch_norm, use_spectral_norm,
                                                 use_layer_norm, use_instance_norm, scale_factor,
                                                 dataset_name=dataset_name)
        self.encoder = self.enc_dec_model.encoder
        self.decoder = self.enc_dec_model.decoder
        self.N = torch.distributions.Normal(0, 1)
        self.bernoulli_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.tc = 0
        self.kl_loss = 0
        self.reconstruction_loss = 0
        self.latent_loss = 0
        self.combined_loss = 0

    def forward(self, x: TensorType["batch", "num_channels", "x", "y"],
                step: int
                ) -> TensorType:
        z_mean, z_logvar = self.encoder(x)
        z_var = torch.exp(z_logvar / 2)
        z_sampled = z_mean + z_var * self.N.sample(z_mean.shape)
        reconstruction = self.decoder(z_sampled)
        self.kl_loss = compute_gaussian_kl(z_mean, z_logvar).mean()
        self.tc = total_correlation(z_sampled, z_mean, z_logvar, 0)
        # The sigmoid activation gets applied in the BCEWithLogitsLoss
        per_sample_reconstruction_loss = torch.sum(self.bernoulli_loss(reconstruction, x), dim=[1, 2, 3])
        self.reconstruction_loss = torch.mean(per_sample_reconstruction_loss)

        return reconstruction, z_sampled, self.combine_loss(step)

    def combine_loss(self, step: int):
        self.latent_loss = self.beta * self.kl_loss
        self.combined_loss = self.latent_loss + self.reconstruction_loss
        return self.combined_loss

    def reconstruct(self, x: TensorType["batch", "num_channels", "x", "y"]
                ) -> (TensorType["batch", "num_channels", "x", "y"], TensorType["batch", "z_dim"]):
        z_mean, z_logvar = self.encoder(x)
        z_var = torch.exp(z_logvar / 2)
        z_sampled = z_mean + z_var * self.N.sample(z_mean.shape)
        reconstruction = self.decoder(z_sampled).sigmoid()
        return reconstruction, z_sampled

    def reconstruct_deterministic(self, x: TensorType["batch", "num_channels", "x", "y"]
                ) -> (TensorType["batch", "num_channels", "x", "y"], TensorType["batch", "z_dim"]):
        z_mean, z_logvar = self.encoder(x)
        reconstruction = self.decoder(z_mean).sigmoid()
        return reconstruction, z_mean

    def encode(self, x: TensorType["batch", "num_channels", "x", "y"]
                ) -> (TensorType["batch", "output_dim"], TensorType["batch", "output_dim"]):
        return self.encoder(x)

    def decode(self, z: TensorType["batch", "input_dim"]
                ) -> TensorType["batch", "num_channels", "x", "y"]:
        return self.decoder(z).sigmoid()

    def get_enc_layer_weights(self):
        return self.encoder

    @property
    def scale_factor(self):
        return self.encoder.scale_factor

    def to(self: BetaVAE, device: Optional[Union[int, device]] = ..., dtype: Optional[Union[dtype, str]] = ...,
           non_blocking: bool = ...) -> BetaVAE:
        super(BetaVAE, self).to(device)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        return self


class BetaTCVAE(BetaVAE):
    def __init__(self, z_dim: int, num_channels: int, beta: int, use_batch_norm: bool = False,
                 use_spectral_norm: bool = False, use_layer_norm: bool = False,
                 use_instance_norm: bool = False, scale_factor: int = 1,
                 dataset_name = 'dsprites', **kwargs):
        super(BetaTCVAE, self).__init__(z_dim, num_channels, beta, use_batch_norm,
                                        use_spectral_norm, use_layer_norm, use_instance_norm, scale_factor,
                                        dataset_name = dataset_name, **kwargs)

    def combine_loss(self, step: int):
        self.latent_loss = (self.beta - 1.) * self.tc + self.kl_loss
        self.combined_loss = self.latent_loss + self.reconstruction_loss
        return self.combined_loss


class AnnealedVAE(BetaVAE):
    def __init__(self, z_dim: int, num_channels: int, gamma: int, c_max: float, iteration_threshold: int,
                 annealed_loss_degree=4, dataset_name='dsprites',
                 use_batch_norm: bool = False, use_spectral_norm: bool = False, use_layer_norm: bool = False,
                 use_instance_norm: bool = False, scale_factor: int = 1, **kwargs):
        super(AnnealedVAE, self).__init__(z_dim, num_channels, 1,
                                          use_batch_norm=use_batch_norm,
                                          use_spectral_norm=use_spectral_norm,
                                          use_layer_norm=use_layer_norm,
                                          use_instance_norm=use_instance_norm,
                                          scale_factor=scale_factor,
                                          dataset_name=dataset_name,
                                          **kwargs)
        self.gamma = gamma
        self.c_max = c_max
        self.iteration_threshold = iteration_threshold
        self.loss_degree = annealed_loss_degree

    def anneal(self, step):
        """Anneal function for anneal_vae (https://arxiv.org/abs/1804.03599).

        Args:
          c_max: Maximum capacity.
          step: Current step.
          iteration_threshold: How many iterations to reach c_max.

        Returns:
          Capacity annealed linearly until c_max.
        """
        return np.min((self.c_max, self.c_max * step / self.iteration_threshold))

    def combine_loss(self, step: int):
        if self.loss_degree == 0:
            self.latent_loss = self.gamma * torch.abs(self.kl_loss - self.anneal(step))
        else:
            self.latent_loss = self.gamma * torch.pow(self.kl_loss - self.anneal(step), self.loss_degree)
        self.combined_loss = self.latent_loss + self.reconstruction_loss
        return self.combined_loss

