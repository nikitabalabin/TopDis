# This code is based on the code from https://github.com/besterma/dava

import os
from pathlib import Path
import torch
import torchvision.utils
import numpy as np
import random

# from dava.models import BetaVAE, Discriminator
from models import BetaVAE, Discriminator


def log_images(batch_size, current_step, device, reconstruct_images, vae, writer,
               uniform_sampling=False, gaussian_sampling=False):
    x_t, z_t = vae.reconstruct(reconstruct_images)
    rec_grid = torchvision.utils.make_grid(x_t.detach().cpu())
    writer.add_image('reconstructions', rec_grid, current_step)
    if uniform_sampling:
        random_state = np.random.RandomState(current_step)
        means = z_t.detach().cpu().numpy()
        random_code = (np.max(means, axis=0) - np.min(means, axis=0)) * random_state.random_sample(
            (batch_size, vae.z_dim)) + np.min(means,
                                              axis=0)
        z_hat = torch.tensor(random_code, dtype=torch.float).to(device)
    elif gaussian_sampling:
        random_state = np.random.RandomState(current_step)
        random_code = random_state.uniform(0,1,(batch_size, vae.z_dim))
        z_hat = torch.tensor(random_code, dtype=torch.float).to(device)
    else:
        z_hat = torch.zeros((batch_size, vae.z_dim), device=device)
        for j in range(vae.z_dim):
            indices = torch.randint(0, batch_size, (batch_size,))
            z_hat[:, j] = z_t[indices, j].detach()

    x_sampled = vae.decode(z_hat)
    samp_grid = torchvision.utils.make_grid(x_sampled.detach().cpu())
    writer.add_image('samples', samp_grid, current_step)

def save_checkpoint(vae: BetaVAE, disc: Discriminator = None, path=".", iter_num=0):
    torch.save(vae.decoder.state_dict(), os.path.join(path, "dec_{}.pth".format(iter_num)))
    torch.save(vae.encoder.state_dict(), os.path.join(path, "enc_{}.pth".format(iter_num)))
    torch.save(vae.state_dict(), os.path.join(path, "model_{}.pth".format(iter_num)))
    if disc:
        torch.save(disc.state_dict(), os.path.join(path, "disc_{}.pth".format(iter_num)))

def store_dict(input_dict: dict, store_path: str):
    with open(store_path, "w") as file:
        for key, value in input_dict.items():
            file.write(f"{key}: {value}\n")


def init_random_state(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed_all(random_seed)
    torch.random.manual_seed(random_seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def smooth_positive_labels(y, factor=0.6):
    return y - (torch.rand(y.shape, device=y.device) * factor)


def smooth_negative_labels(y, factor=0.6):
    return y + torch.rand(y.shape, device=y.device) * factor


def smooth_labels(y):
    if y[0] == 0:
        return smooth_negative_labels(y)
    else:
        return smooth_positive_labels(y)


def prepare_store_path(store_path: str, dataset_path: str, num_channels: int, z_dim: int):
    Path(store_path).mkdir(parents=True, exist_ok=True)
    Path(store_path, "results", "gin").mkdir(parents=True, exist_ok=True)
    dataset_name = ""
    if dataset_path.endswith("dsprites.npz"):
        dataset_name = "dsprites_full"
    elif dataset_path.endswith("3dshapes.npz"):
        dataset_name = "shapes3d"
    elif dataset_path.endswith("celeba.npz"):
        dataset_name = 'celeba'
    elif dataset_path.endswith("mpi3d_toy.npz"):
        dataset_name = 'mpi3d_toy'
    gin_string = f"dataset.name='{dataset_name}'\n" \
                 f"VAE.num_channels = {num_channels}\n" \
                 f"VAE.z_dim = {z_dim}\n"
    with open(os.path.join(store_path, "results", "gin", "train.gin"), "w") as gin_file:
        gin_file.write(gin_string)