# This code is based on the code from https://github.com/1Konny/FactorVAE

"""solver.py"""

import os
import visdom
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from utils import DataGather, mkdirs, grid2gif
from ops import recon_loss, kl_divergence, permute_dims
from model import FactorVAE1, FactorVAE2, FactorVAE3
from dataset import return_data
import csv
from rtd_regularizer import *

def get_scalar_product(model, recon_grads):
    sp = 0
    for i, p in enumerate(model.parameters()):
        if p.grad is not None:
            g_recon = recon_grads[i]
            g_rtd = p.grad - g_recon
            sp += torch.sum(g_rtd * g_recon)
    return sp

def make_step(model, optimizer, recon_loss, rtd_loss, other_losses):
    optimizer.zero_grad()
    recon_loss.backward(retain_graph=True)
    recon_grads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
    rtd_loss.backward(retain_graph=True)

    num = get_scalar_product(model, recon_grads)
    
    if num < 0:
        denom = 0
        for g_recon in recon_grads:
            denom += torch.sum(g_recon * g_recon)
            
        for i, p in enumerate(model.parameters()):
            if p.grad is not None:
                g_recon = recon_grads[i]
                p.grad = p.grad - num / denom * g_recon
            
    sp_unort = num
    sp_ort = get_scalar_product(model, recon_grads)
    
    for loss in other_losses:
        loss.backward(retain_graph=True)
            
    optimizer.step()
    
    return sp_unort.item(), sp_ort.item()

class Solver(object):
    def __init__(self, args):
        # Misc
        use_cuda = args.cuda and torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'
        self.name = args.name
        self.max_iter = int(args.max_iter)
        self.print_iter = args.print_iter
        self.global_iter = 0
        self.pbar = tqdm(total=self.max_iter)

        # Data
        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.data_loader = return_data(args)

        # Networks & Optimizers
        self.z_dim = args.z_dim
        self.beta = args.beta
        print ('Beta: {}'.format(self.beta))

        self.lr_VAE = args.lr_VAE
        self.beta1_VAE = args.beta1_VAE
        self.beta2_VAE = args.beta2_VAE

        if args.dataset == 'dsprites':
            self.VAE = FactorVAE1(self.z_dim, arch_type=2).to(self.device)
            self.nc = 1
        elif args.dataset in ['3dshapes', 'mpi3d_complex']:
            self.VAE = FactorVAE2(self.z_dim).to(self.device)
            self.nc = 3
        elif args.dataset == '3dfaces':
            self.VAE = FactorVAE3(self.z_dim).to(self.device)
            self.nc = 1
        self.optim_VAE = optim.Adam(self.VAE.parameters(), lr=self.lr_VAE,
                                    betas=(self.beta1_VAE, self.beta2_VAE))

        self.nets = [self.VAE]
        
        # RTD regularization
        self.use_rtd = args.use_rtd
        self.lp = args.lp
        self.q_normalize = args.q_normalize
        self.gamma_rtd = args.gamma_rtd
        self.sample_based = args.sample_based
        self.delay_iter = args.delay_iter
        self.weightnorm_sampler = args.weightnorm_sampler
        self.rtd_reg = RTDRegularizer(self.lp, self.q_normalize, self.sample_based, self.weightnorm_sampler)
        self.orthogonalize = args.orthogonalize
        self.orthogonalize_all = args.orthogonalize_all
        if self.use_rtd:
            print ('Use RTD regularization with lp = {}, q_normalize = {}, gamma_rtd = {}, sample_based = {}, delay_iter = {}, weightnorm_sampler = {}, orthogonalize = {}, orthogonalize_all = {}'.format(self.lp, self.q_normalize, self.gamma_rtd, self.sample_based, self.delay_iter, self.weightnorm_sampler, self.orthogonalize, self.orthogonalize_all))
        # Visdom
        self.viz_on = args.viz_on
        self.win_id = dict(recon='win_recon', kld='win_kld')
        self.line_gather = DataGather('iter', 'recon', 'kld')
        self.image_gather = DataGather('true', 'recon')
        self.viz_ta_iter = args.viz_ta_iter
        if self.viz_on:
            self.viz_port = args.viz_port
            self.viz = visdom.Visdom(port=self.viz_port)
            self.viz_ll_iter = args.viz_ll_iter
            self.viz_la_iter = args.viz_la_iter
            self.viz_ra_iter = args.viz_ra_iter
            if not self.viz.win_exists(env=self.name+'/lines'):
                self.viz_init()
                
        print(self.VAE.encode[0].weight[0,0].data)
        print(self.VAE.decode[0].weight[0,0].data)
        # Checkpoint
        self.ckpt_dir = os.path.join(args.ckpt_dir, args.name)
        self.ckpt_save_iter = args.ckpt_save_iter
        mkdirs(self.ckpt_dir)
        if args.ckpt_load:
            self.load_checkpoint(args.ckpt_load)

        # Output(latent traverse GIF)
        self.output_dir = os.path.join(args.output_dir, args.name)
        self.output_save = args.output_save
        mkdirs(self.output_dir)

    def train(self):
        self.net_mode(train=True)

        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        
        columns = ['iter', 'vae_recon_loss', 'vae_kld_loss']
        if self.use_rtd:
            columns.append('rtd_loss')
            columns.append('i')
            if self.orthogonalize or self.orthogonalize_all:
                columns.append('sp_unort')
                columns.append('sp_ort')
        stats_file_path = os.path.join(self.output_dir, 'stats.csv')
        if not os.path.exists(stats_file_path):
            with open(stats_file_path, mode='w') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=columns)
                writer.writeheader()

        out = False
        while not out:
            for x_true1, x_true2 in self.data_loader:
                self.global_iter += 1
                self.pbar.update(1)

                x_true1 = x_true1.to(self.device)
                x_recon, mu, logvar, z = self.VAE(x_true1)
                vae_recon_loss = recon_loss(x_true1, x_recon)
                vae_kld = kl_divergence(mu, logvar)

                vae_loss = vae_recon_loss + self.beta*vae_kld
                
                if self.use_rtd and self.global_iter > self.delay_iter:
                    x_true2 = x_true2.to(self.device)
                    i, rtd_loss = self.rtd_reg.compute_reg(self.VAE, x_true2)
                    vae_loss += self.gamma_rtd * rtd_loss
                    
                if self.use_rtd and self.global_iter > self.delay_iter and self.orthogonalize:
                    sp_unort, sp_ort = make_step(model=self.VAE, optimizer=self.optim_VAE,
                                                 recon_loss=vae_recon_loss,
                                                 rtd_loss=self.gamma_rtd*rtd_loss,
                                                 other_losses=[self.beta*vae_kld])
                elif self.use_rtd and self.global_iter > self.delay_iter and self.orthogonalize_all:
                    sp_unort, sp_ort = make_step(model=self.VAE, optimizer=self.optim_VAE,
                                                 recon_loss=vae_recon_loss + self.beta*vae_kld,
                                                 rtd_loss=self.gamma_rtd*rtd_loss,
                                                 other_losses=[])  
                else:
                    self.optim_VAE.zero_grad()
                    vae_loss.backward(retain_graph=True)
                    self.optim_VAE.step()
                
                with open(stats_file_path, mode='a') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=columns)
                    row_dict = {'iter': self.global_iter,
                                'vae_recon_loss': vae_recon_loss.item(),
                                'vae_kld_loss': vae_kld.item(),
                               }
                    if self.use_rtd and self.global_iter > self.delay_iter:
                        row_dict['rtd_loss'] = rtd_loss.item()
                        row_dict['i'] = i
                        if self.orthogonalize or self.orthogonalize_all:
                            row_dict['sp_unort'] = sp_unort
                            row_dict['sp_ort'] = sp_ort
                    writer.writerow(row_dict)

                if self.global_iter%self.print_iter == 0:
                    out_line = '[{}] vae_recon_loss:{:.3f} vae_kld:{:.3f}'.format(
                        self.global_iter, vae_recon_loss.item(), vae_kld.item())
                    if self.use_rtd and self.global_iter > self.delay_iter:
                        out_line += ' RTD_loss:{:.3f}'.format(rtd_loss.item())
                    self.pbar.write(out_line)

                if self.global_iter%self.ckpt_save_iter == 0:
                    self.save_checkpoint(self.global_iter)

                if self.viz_on and (self.global_iter%self.viz_ll_iter == 0):
                    self.line_gather.insert(iter=self.global_iter,
                                            recon=vae_recon_loss.item(),
                                            kld=vae_kld.item(),
                                           )

                if self.viz_on and (self.global_iter%self.viz_la_iter == 0):
                    self.visualize_line()
                    self.line_gather.flush()

                if self.viz_on and (self.global_iter%self.viz_ra_iter == 0):
                    self.image_gather.insert(true=x_true1.data.cpu(),
                                             recon=F.sigmoid(x_recon).data.cpu())
                    self.visualize_recon()
                    self.image_gather.flush()

                if self.global_iter%self.viz_ta_iter == 0:
                    if self.dataset.lower() == '3dchairs':
                        self.visualize_traverse(limit=2, inter=0.5)
                    else:
                        self.visualize_traverse(limit=3, inter=2/3)

                if self.global_iter >= self.max_iter:
                    out = True
                    break

        self.pbar.write("[Training Finished]")
        self.pbar.close()

    def visualize_recon(self):
        data = self.image_gather.data
        true_image = data['true'][0]
        recon_image = data['recon'][0]

        true_image = make_grid(true_image)
        recon_image = make_grid(recon_image)
        sample = torch.stack([true_image, recon_image], dim=0)
        self.viz.images(sample, env=self.name+'/recon_image',
                        opts=dict(title=str(self.global_iter)))

    def visualize_line(self):
        data = self.line_gather.data
        iters = torch.Tensor(data['iter'])
        recon = torch.Tensor(data['recon'])
        kld = torch.Tensor(data['kld'])

        self.viz.line(X=iters,
                      Y=recon,
                      env=self.name+'/lines',
                      win=self.win_id['recon'],
                      update='append',
                      opts=dict(
                        xlabel='iteration',
                        ylabel='reconstruction loss',))
        self.viz.line(X=iters,
                      Y=kld,
                      env=self.name+'/lines',
                      win=self.win_id['kld'],
                      update='append',
                      opts=dict(
                        xlabel='iteration',
                        ylabel='kl divergence',))

    def visualize_traverse(self, limit=3, inter=2/3, loc=-1):
        self.net_mode(train=False)

        decoder = self.VAE.decode
        encoder = self.VAE.encode
        interpolation = torch.arange(-limit, limit+0.1, inter)

        random_img = self.data_loader.dataset.__getitem__(0)[1]
        random_img = random_img.to(self.device).unsqueeze(0)
        random_img_z = encoder(random_img)[:, :self.z_dim]

        if self.dataset.lower() == 'dsprites':
            fixed_idx1 = 87040 # square
            fixed_idx2 = 332800 # ellipse
            fixed_idx3 = 578560 # heart

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            Z = {'fixed_square':fixed_img_z1, 'fixed_ellipse':fixed_img_z2,
                 'fixed_heart':fixed_img_z3, 'random_img':random_img_z}

        elif self.dataset.lower() == 'celeba':
            fixed_idx1 = 191281 # 'CelebA/img_align_celeba/191282.jpg'
            fixed_idx2 = 143307 # 'CelebA/img_align_celeba/143308.jpg'
            fixed_idx3 = 101535 # 'CelebA/img_align_celeba/101536.jpg'
            fixed_idx4 = 70059  # 'CelebA/img_align_celeba/070060.jpg'

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            fixed_img4 = self.data_loader.dataset.__getitem__(fixed_idx4)[0]
            fixed_img4 = fixed_img4.to(self.device).unsqueeze(0)
            fixed_img_z4 = encoder(fixed_img4)[:, :self.z_dim]

            Z = {'fixed_1':fixed_img_z1, 'fixed_2':fixed_img_z2,
                 'fixed_3':fixed_img_z3, 'fixed_4':fixed_img_z4,
                 'random':random_img_z}

        elif self.dataset.lower() == '3dchairs':
            fixed_idx1 = 40919 # 3DChairs/images/4682_image_052_p030_t232_r096.png
            fixed_idx2 = 5172  # 3DChairs/images/14657_image_020_p020_t232_r096.png
            fixed_idx3 = 22330 # 3DChairs/images/30099_image_052_p030_t232_r096.png

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            Z = {'fixed_1':fixed_img_z1, 'fixed_2':fixed_img_z2,
                 'fixed_3':fixed_img_z3, 'random':random_img_z}
        else:
            fixed_idx = 0
            fixed_img = self.data_loader.dataset.__getitem__(fixed_idx)[0]
            fixed_img = fixed_img.to(self.device).unsqueeze(0)
            fixed_img_z = encoder(fixed_img)[:, :self.z_dim]

            random_z = torch.rand(1, self.z_dim, 1, 1, device=self.device)

            Z = {'fixed_img':fixed_img_z, 'random_img':random_img_z, 'random_z':random_z}

        gifs = []
        for key in Z:
            z_ori = Z[key]
            samples = []
            for row in range(self.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:
                    z[:, row] = val
                    sample = F.sigmoid(decoder(z)).data
                    samples.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples, dim=0).cpu()
            title = '{}_latent_traversal(iter:{})'.format(key, self.global_iter)
#             self.viz.images(samples, env=self.name+'/traverse',
#                             opts=dict(title=title), nrow=len(interpolation))

        if self.output_save:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            mkdirs(output_dir)
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(Z), self.z_dim, len(interpolation), self.nc, 64, 64).transpose(1, 2)
            for i, key in enumerate(Z.keys()):
                for j, val in enumerate(interpolation):
                    save_image(tensor=gifs[i][j].cpu(),
                               fp=os.path.join(output_dir, '{}_{}.jpg'.format(key, j)),
                               nrow=self.z_dim, pad_value=1)

                grid2gif(str(os.path.join(output_dir, key+'*.jpg')),
                         str(os.path.join(output_dir, key+'.gif')), delay=10)

        self.net_mode(train=True)

    def viz_init(self):
        zero_init = torch.zeros([1])
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env=self.name+'/lines',
                      win=self.win_id['recon'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='reconstruction loss',))
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env=self.name+'/lines',
                      win=self.win_id['kld'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='kl divergence',))

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise ValueError('Only bool type is supported. True|False')

        for net in self.nets:
            if train:
                net.train()
            else:
                net.eval()

    def save_checkpoint(self, ckptname='last', verbose=True):
        model_states = {'VAE':self.VAE.state_dict()}
        optim_states = {'optim_VAE':self.optim_VAE.state_dict()}
        states = {'iter':self.global_iter,
                  'model_states':model_states,
                  'optim_states':optim_states}

        filepath = os.path.join(self.ckpt_dir, str(ckptname))
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        if verbose:
            self.pbar.write("=> saved checkpoint '{}' (iter {})".format(filepath, self.global_iter))

    def load_checkpoint(self, ckptname='last', verbose=True):
        if ckptname == 'last':
            ckpts = os.listdir(self.ckpt_dir)
            if not ckpts:
                if verbose:
                    self.pbar.write("=> no checkpoint found")
                return

            ckpts = [int(ckpt) for ckpt in ckpts]
            ckpts.sort(reverse=True)
            ckptname = str(ckpts[0])

        filepath = os.path.join(self.ckpt_dir, ckptname)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)

            self.global_iter = checkpoint['iter']
            self.VAE.load_state_dict(checkpoint['model_states']['VAE'])
            self.optim_VAE.load_state_dict(checkpoint['optim_states']['optim_VAE'])
            self.pbar.update(self.global_iter)
            if verbose:
                self.pbar.write("=> loaded checkpoint '{} (iter {})'".format(filepath, self.global_iter))
        else:
            if verbose:
                self.pbar.write("=> no checkpoint found at '{}'".format(filepath))
