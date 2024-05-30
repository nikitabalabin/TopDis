# This code is based on the code from https://github.com/besterma/dava

import argparse
import os
import gin
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import numpy.typing as npt

from models import AnnealedVAE, Discriminator
from utils import *

from tqdm import tqdm

from rtd_regularizer import *
import csv
import h5py

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=150000)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--num_channels", type=int, default=1)
    parser.add_argument("--store_path", type=str)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--z_dim", type=int, default=10)
    parser.add_argument("--disc_weight", type=float, default=1)
    parser.add_argument("--random_seed", type=int, default=0)
    # using the default values of c_max and iteration_threshold leads to deltaC of 4e-5
    parser.add_argument("--c_max", type=float, default=8)
    parser.add_argument("--iteration_threshold", type=int, default=100000 * 128)
    parser.add_argument("--gamma", type=int, default=1000)
    parser.add_argument("--batch_norm", action='store_true')
    parser.add_argument("--spectral_norm", action='store_true')
    parser.add_argument("--layer_norm", action='store_true')
    parser.add_argument("--instance_norm", action='store_true')
    parser.add_argument("--disc_batch_norm", action='store_true')
    parser.add_argument("--disc_spectral_norm", action='store_true')
    parser.add_argument("--disc_layer_norm", action='store_true')
    parser.add_argument("--disc_instance_norm", action='store_true')
    parser.add_argument("--disc_learning_rate", type=float, default=0.0001)
    parser.add_argument("--use_mixed_batches", action='store_true')
    parser.add_argument("--max_grad_norm", type=float, default=1.)
    parser.add_argument("--disc_max_grad_norm", type=float, default=1.)
    parser.add_argument("--disc_decoder_weight", type=float, default=0.)
    parser.add_argument("--constant_disc_weight", action='store_true')
    parser.add_argument("--constant_dec_disc_weight", action='store_true')
    parser.add_argument("--model_scale_factor", type=int, default=1)
    parser.add_argument("--disc_scale_factor", type=int, default=1)
    parser.add_argument("--start_kl_step", type=int, default=0)
    parser.add_argument("--disc_confidence_value", type=float, default=0.6)
    parser.add_argument("--annealed_loss_degree", type=int, default=4)
    parser.add_argument("--gaussian_sampling", action='store_true')
    parser.add_argument("--permutation_sampling", action='store_true')
    parser.add_argument("--uniform_sampling", action='store_true')

    ### TopDis specific arguments
    parser.add_argument('--use_topdis', action='store_true', help='use additional topdis loss')
    parser.add_argument('--lp', default=1, type=int, help='norm to compute topological features')
    parser.add_argument('--q_normalize', action='store_true', help='whether to use quantile normalization')
    parser.add_argument('--gamma_topdis', default=1, type=float, help='gamma hyperparameter for topdis loss')
    parser.add_argument('--sample_based', action='store_true', help='compute sample based topdis loss')
    parser.add_argument('--weightnorm_sampler', action='store_true', help='whether to sample dim according to weight norm')
    parser.add_argument('--delay_iter', default=0, type=int, help='number of iterations to wait before optimize topdis loss')
    ###

    args = parser.parse_args()

    init_random_state(args.random_seed)
    dataset_path = args.dataset_path
    num_channels = args.num_channels
    print("Loading Dataset")
    if 'dsprites' in dataset_path:
        dataset = np.load(dataset_path)["imgs"]
        dataset = np.expand_dims(dataset, -1)
        dataset_name = 'dsprites'
    elif 'real3d_complicated_shapes' in dataset_path:
        dataset = np.load(dataset_path)["images"]
        dataset_name = 'mpi3d'
    elif '3dshapes' in dataset_path:
        dataset = h5py.File(dataset_path, 'r')['images'][()]
        dataset_name = '3dshapes'
    elif 'face' in dataset_path:
        dataset = torch.load(dataset_path, map_location='cpu')
        dataset = dataset.flatten(0, -3).unsqueeze(-1)
        dataset = dataset.numpy()
        print ('3d faces shape ', dataset.shape)
        dataset_name = '3dfaces'
    print (dataset_name, dataset.shape)
    print("Finished loading Dataset")
    device = torch.device(args.device)
    batch_size = args.batch_size
    num_steps = args.num_steps * batch_size
    learning_rate = args.learning_rate
    z_dim = args.z_dim
    # beta = args.beta
    disc_weight = args.disc_weight
    store_path = args.store_path
    c_max = args.c_max
    gamma = args.gamma
    use_batch_norm = args.batch_norm
    use_spectral_norm = args.spectral_norm
    use_layer_norm = args.layer_norm
    use_instance_norm = args.instance_norm
    disc_use_bn = args.disc_batch_norm
    disc_use_sn = args.disc_spectral_norm
    disc_use_instance_norm = args.disc_instance_norm
    disc_learning_rate = args.disc_learning_rate
    use_mixed_batches = args.use_mixed_batches
    max_grad_norm = args.max_grad_norm
    disc_max_grad_norm = args.disc_max_grad_norm
    use_uniform_sampling = args.uniform_sampling
    use_gaussian_sampling = args.gaussian_sampling
    permutation_sampling = args.permutation_sampling
    disc_decoder_weight = args.disc_decoder_weight
    constant_disc_weight = args.constant_disc_weight
    constant_dec_disc_weight = args.constant_dec_disc_weight
    model_scale_factor = args.model_scale_factor
    disc_scale_factor = args.disc_scale_factor
    start_kl_step = args.start_kl_step
    disc_confidence_value = args.disc_confidence_value
    annealed_loss_degree = args.annealed_loss_degree
    # prepare_store_path(store_path, dataset_path, num_channels, z_dim, beta)
    prepare_store_path(store_path, dataset_path, num_channels, z_dim)
    store_dict(args.__dict__, os.path.join(store_path, "parameters.txt"))

    vae = AnnealedVAE(z_dim=z_dim, num_channels=num_channels,
                      c_max=c_max, gamma=gamma, iteration_threshold=args.iteration_threshold,
                      use_batch_norm=use_batch_norm, use_spectral_norm=use_spectral_norm,
                      use_layer_norm=use_layer_norm, use_instance_norm=use_instance_norm,
                      scale_factor=model_scale_factor, annealed_loss_degree=annealed_loss_degree,
                      dataset_name=dataset_name).to(device)
    print(vae.encoder.encode[0].weight[0,0].data)
    print(vae.decoder.decode[0].weight[0,0].data)

    disc = Discriminator(1, num_channels, use_spectral_norm=disc_use_sn, use_batch_norm=disc_use_bn,
                         use_layer_norm=use_layer_norm, use_instance_norm=disc_use_instance_norm,
                         scale_factor=disc_scale_factor, dataset_name=dataset_name).to(device)

    # TopDis loss
    use_topdis = args.use_topdis
    lp = args.lp
    q_normalize = args.q_normalize
    gamma_topdis = args.gamma_topdis
    sample_based = args.sample_based
    weightnorm_sampler = args.weightnorm_sampler
    delay_iter = args.delay_iter
    topdis_loss = None
    if use_topdis:
        topdis_loss = RTDRegularizer(lp, q_normalize, sample_based, weightnorm_sampler)
        print ('Use TopDis loss with lp = {}, q_normalize = {}, gamma_topdis = {}, sample_based = {}, weightnorm_sampler = {}, delay_iter = {}'.format(lp, q_normalize, gamma_topdis, sample_based, weightnorm_sampler, delay_iter))

    train_dava(vae, disc, dataset, device, store_path=store_path, batch_size=batch_size, num_steps=num_steps,
               learning_rate=learning_rate, disc_weight=disc_weight,
               dynamic=True, max_grad_norm=max_grad_norm, uniform_sampling=use_uniform_sampling,
               gaussian_sampling=use_gaussian_sampling,
               update_decoder=disc_decoder_weight, disc_max_grad_norm=disc_max_grad_norm,
               disc_learning_rate=disc_learning_rate, use_mixed_batches=use_mixed_batches,
               constant_disc_weight=constant_disc_weight,
               constant_dec_disc_weight=constant_dec_disc_weight,
               start_kl_step=start_kl_step, disc_confidence_value=disc_confidence_value,
               permutation=permutation_sampling,
               use_topdis=use_topdis, topdis_loss=topdis_loss, gamma_topdis=gamma_topdis, delay_iter=delay_iter)


def train_dava(vae: AnnealedVAE, disc: Discriminator, data: npt.ArrayLike, device, store_path: str, batch_size=32,
               num_steps=5, learning_rate=1e-4, disc_weight=1, dynamic=False, max_grad_norm=1.,
               uniform_sampling=False, gaussian_sampling=False, update_decoder=False,
               disc_max_grad_norm=1., disc_learning_rate=0.0001, use_mixed_batches=True, constant_disc_weight=False,
               constant_dec_disc_weight=True, start_kl_step=0, disc_confidence_value=0.5, permutation=False,
               use_topdis=False, topdis_loss=None, gamma_topdis=0., delay_iter=0):
    assert not (uniform_sampling and gaussian_sampling), "Cant do uniform and gaussian sampling at the same time"
    data_loader = DataLoader(data, batch_size=batch_size, pin_memory=True,
                             drop_last=True, shuffle=True, num_workers=0)
    opt_enc = optim.Adam(vae.encoder.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
    opt_dec = optim.Adam(vae.decoder.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
    opt_disc = optim.Adam(disc.parameters(), lr=disc_learning_rate, betas=(0.9, 0.999), eps=1e-08)
    loss_disc = nn.BCEWithLogitsLoss(reduction="sum")

    rec_label = 1
    sampled_label = 0

    num_correct = 0
    num_total = 0

    label_rec = torch.full((batch_size,), rec_label, dtype=torch.float32, device=device)
    label_sampled = torch.full((batch_size,), sampled_label, dtype=torch.float32, device=device)

    current_kl_step = start_kl_step

    columns = ['iter', 'vae_loss', 'disc_loss', 'disc_loss_vae_enc', 'disc_loss_vae_dec', 'current_kl_step']
    if use_topdis:
        columns.append('topdis_loss')
        columns.append('i')
    stats_file_path = os.path.join(store_path, 'stats.csv')
    if not os.path.exists(stats_file_path):
        with open(stats_file_path, mode='w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            writer.writeheader()

    current_step = 0
    end_reached = False
    epoch_num = 0
    while not end_reached:
        for i, x in tqdm(enumerate(data_loader)):
            current_step += batch_size
            if current_step > num_steps:
                end_reached = True
                break

            ###########################################################################################################
            ####                    Train VAE                                                                       ###
            ###########################################################################################################
            vae.train()
            # train vae reconstruction loss
            vae.zero_grad()
            x = x.to(device)
            x = x.type(torch.float32)
            if x.max() > 1:
                x = x / 255.
            x = torch.permute(x, (0, 3, 1, 2))

            x_t, _, vae_loss = vae(x, current_kl_step)


            ### TopDis loss
            if (use_topdis) and (topdis_loss is not None) and (current_step // batch_size > delay_iter):
                i, topdis_loss_value = topdis_loss.compute_reg(vae, x, current_kl_step)
                vae_loss += gamma_topdis * topdis_loss_value
            ###

            vae_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=max_grad_norm)
            opt_enc.step()
            opt_dec.step()

            ###########################################################################################################
            ####                    Train Disc                                                                      ###
            ###########################################################################################################
            disc.zero_grad()
            vae.eval()
            disc.train()

            x_rec, z_t = vae.reconstruct(x)
            if uniform_sampling:
                random_state = np.random.RandomState(current_step)
                means = z_t.detach().cpu().numpy()
                random_code = (np.max(means, axis=0) - np.min(means, axis=0)) * random_state.random_sample(
                    (batch_size, vae.z_dim)) + np.min(means,
                                                      axis=0)
                z_hat = torch.tensor(random_code, dtype=torch.float).to(device)
            elif gaussian_sampling:
                random_state = np.random.RandomState(current_step)
                random_code = random_state.normal(0, 1, (batch_size, vae.z_dim))
                z_hat = torch.tensor(random_code, dtype=torch.float).to(device)
            elif permutation:
                z_hat = torch.zeros((batch_size, vae.z_dim), device=device)
                for j in range(vae.z_dim):
                    indices = torch.randperm(batch_size)
                    z_hat[:, j] = z_t[indices, j].detach()
            else:
                z_hat = torch.zeros((batch_size, vae.z_dim), device=device)
                for j in range(vae.z_dim):
                    indices = torch.randint(0, batch_size, (batch_size,))
                    z_hat[:, j] = z_t[indices, j].detach()

            x_sampled = vae.decode(z_hat)

            if use_mixed_batches:
                middle = batch_size // 2
                x1 = torch.cat((x_rec[:middle], x_sampled[:middle]))
                l1 = torch.cat((smooth_positive_labels(label_rec, disc_confidence_value)[:middle],
                                smooth_negative_labels(label_sampled, disc_confidence_value)[:middle]))
                l_t_1 = torch.cat((label_rec[:middle], label_sampled[:middle]))
                x2 = torch.cat((x_rec[middle:], x_sampled[middle:]))
                l2 = torch.cat((smooth_positive_labels(label_rec, disc_confidence_value)[middle:],
                                smooth_negative_labels(label_sampled, disc_confidence_value)[middle:]))
                l_t_2 = torch.cat((label_rec[middle:], label_sampled[middle:]))
                training_images = [(x1, l1, l_t_1), (x2, l2, l_t_2)]
            else:
                training_images = [(x_rec, smooth_positive_labels(label_rec), label_rec),
                                   (x_sampled, smooth_negative_labels(label_sampled), label_sampled)]

            loss = torch.zeros((1)).to(device)
            batch_correct = 0
            for x, l, l_t in training_images:
                output = disc(x.detach()).view(-1)
                pred_y = output.sigmoid() >= disc_confidence_value
                correct_rec = torch.sum(pred_y == l_t).detach().to("cpu").numpy()
                num_correct += correct_rec
                batch_correct += correct_rec
                num_total += pred_y.size(0)
                err = loss_disc(output, l)
                loss += err
            loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=disc_max_grad_norm)
            opt_disc.step()

            ###########################################################################################################
            ####                    Train VAE with Disc Loss                                                        ###
            ###########################################################################################################

            accuracy_based_weight = 1
            if dynamic:
                batch_accuracy = batch_correct / (2 * batch_size)
                accuracy_based_weight = np.max(((batch_accuracy - 0.50) * 100, 0))
                if batch_accuracy <= 0.501:
                    current_kl_step += batch_size // 2
                elif batch_accuracy <= 0.51:
                    pass
                else:
                    current_kl_step = current_kl_step - batch_size // 2
                if constant_disc_weight:
                    effective_disc_weight = disc_weight
                else:
                    effective_disc_weight = accuracy_based_weight * disc_weight
            else:
                effective_disc_weight = disc_weight

            disc.eval()
            vae.train()
            # train vae with adversarial loss on reconstructions
            output = disc(x_rec).view(-1)
            raw_loss = loss_disc(output, label_sampled)

            # factor should be > 0, as we want the disc to classify the reconstructed images as sampled
            factor = effective_disc_weight
            err_enc = factor * raw_loss
            err_enc.backward(inputs=list(vae.encoder.parameters()))

            torch.nn.utils.clip_grad_norm_(vae.encoder.parameters(), max_norm=max_grad_norm)

            if update_decoder > 0.:
                x_sampled = vae.decode(z_hat)
                output = disc(x_sampled).view(-1)
                raw_loss = loss_disc(output, label_rec)
                if constant_dec_disc_weight or not dynamic:
                    factor = update_decoder
                else:
                    factor = np.min((update_decoder * accuracy_based_weight, 0.001))
                err_dec = factor * raw_loss
                err_dec.backward(inputs=list(vae.decoder.parameters()))
                opt_dec.step()

            opt_enc.step()

            with open(stats_file_path, mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=columns)
                row_dict = {'iter': current_step / batch_size,
                            'vae_loss': vae_loss.item(),
                            'disc_loss': loss.item(),
                            'disc_loss_vae_enc': err_enc.item(),
                            'current_kl_step': current_kl_step,
                           }
                if update_decoder > 0:
                    row_dict['disc_loss_vae_dec'] = err_dec.item()
                if (use_topdis) and (current_step // batch_size > delay_iter):
                    row_dict['topdis_loss'] = topdis_loss_value.item()
                    row_dict['i'] = i
                writer.writerow(row_dict)

        print(f"Finished an epoch")
        # save_checkpoint(vae, disc, store_path)
        if epoch_num % 10 == 0:
            save_checkpoint(vae, disc, store_path, current_step // batch_size)
        epoch_num += 1

    print("Finished Training")
    # save_checkpoint(vae, disc, store_path)
    save_checkpoint(vae, disc, store_path, current_step // batch_size)


if __name__ == '__main__':
    main()
