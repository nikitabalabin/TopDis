"""solver.py"""


import torch
# torch.cuda.set_device(0)
import warnings
warnings.filterwarnings("ignore")

import os
from tqdm import tqdm
import visdom

import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

from utils import cuda, grid2gif
from model import BetaVAE_H, BetaVAE_B, BetaVAE_HL
from dataset import return_data
# from I_PID import PIDControl
from P_PID import PIDControl

import matplotlib.pyplot as plt

from rtd_regularizer import *

def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss
    

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[],
                    recon_loss=[],
                    total_kld=[],
                    dim_wise_kld=[],
                    mean_kld=[],
                    mu=[],
                    var=[],
                    images=[], beta=[])

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()


class Solver(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.max_iter = args.max_iter
        self.global_iter = 0

        self.z_dim = args.z_dim
        self.beta = args.beta
        self.gamma = args.gamma
        self.C_max = args.C_max
        self.C_max_org = args.C_max
        self.C_stop_iter = args.C_stop_iter
        self.objective = args.objective
        self.model = args.model
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        # self.KL_loss = args.KL_loss
        self.pid_fixed = args.pid_fixed
        self.is_PID = args.is_PID
        self.step_value = args.step_val
        self.C_start = args.C_start
        self.rtd_iter_start = args.delay_iter
        
        if args.dataset.lower() == 'dsprites':
            self.nc = 1
            self.decoder_dist = 'bernoulli'
        elif args.dataset.lower() == 'mpi3d':
            self.nc = 3
            self.decoder_dist = 'bernoulli'
        elif args.dataset.lower() == '3dfaces':
            self.nc = 1
            self.decoder_dist = 'bernoulli'
        elif args.dataset.lower() == '3dshapes':
            self.nc = 3
            self.decoder_dist = 'bernoulli'
        elif args.dataset.lower() == '3dchairs':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'celeba':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        else:
            raise NotImplementedError

        if args.model == 'H':
            net = BetaVAE_H
        elif args.model == 'HL':
            net = BetaVAE_HL
        elif args.model == 'B':
            net = BetaVAE_B
        else:
            raise NotImplementedError('only support model H or B')

        self.net = cuda(net(self.z_dim, self.nc), self.use_cuda)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,
                                    betas=(self.beta1, self.beta2))

        self.viz_name = args.viz_name
        self.viz_port = args.viz_port
        self.viz_on = args.viz_on
        self.win_recon = None
        self.win_beta = None
        self.win_kld = None
        self.win_mu = None
        self.win_var = None
        if self.viz_on:
            self.viz = visdom.Visdom(port=self.viz_port)

        self.ckpt_dir = os.path.join(args.ckpt_dir, args.viz_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        self.save_output = args.save_output
        self.output_dir = os.path.join(args.output_dir, args.viz_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.gather_step = args.gather_step
        self.display_step = args.display_step
        self.save_step = args.save_step

        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.data_loader = return_data(args)
        
        self.gather = DataGather()
        self.gather2 = DataGather()
        
        # RTD regularization
        self.use_rtd = args.use_rtd
        self.lp = args.lp
        self.q_normalize = args.q_normalize
        self.gamma_rtd = args.gamma_rtd
        self.sample_based = args.sample_based
        self.weightnorm_sampler = args.weightnorm_sampler
        self.rtd_reg = None
        if self.use_rtd:
            self.rtd_reg = RTDRegularizer(self.lp, self.q_normalize, self.sample_based, self.weightnorm_sampler)
            print ('Use RTD regularization with lp = {}, q_normalize = {}, gamma_rtd = {}, sample_based = {}, weightnorm_sampler = {}'.format(self.lp, self.q_normalize, self.gamma_rtd, self.sample_based, self.weightnorm_sampler))


    def train(self):
        self.net_mode(train=True)
        self.C_max = Variable(cuda(torch.FloatTensor([self.C_max]), self.use_cuda))
        out = False
        
        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        ## write log to log file
        outfile = os.path.join(self.ckpt_dir, "train.log")
        kl_file = os.path.join(self.ckpt_dir, "train.kl")
        fw_log = open(outfile, "w")
        fw_kl = open(kl_file, "w")
        # fw_kl.write('total KL\tz_dim' + '\n')

        ## init PID control
        PID = PIDControl()
        Kp = 0.01
        Ki = -0.001
        Kd = 0.0
        C = 0.5
        period = 5000
        fw_log.write("Kp:{0:.5f} Ki: {1:.6f} C_iter:{2:.1f} period:{3} step_val:{4:.4f}\n" \
                    .format(Kp, Ki, self.C_stop_iter, period,self.step_value))
        
        while not out:
            for x in self.data_loader:
                self.global_iter += 1
                pbar.update(1)

                x = Variable(cuda(x, self.use_cuda))
                x_recon, mu, logvar = self.net(x)
                recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
                
                if self.is_PID and self.objective == 'H':
                    if self.global_iter%period==0:
                        C += self.step_value
                    if C > self.C_max_org:
                        C = self.C_max_org
                    ## dynamic pid
                    self.beta, _ = PID.pid(C, total_kld.item(), Kp, Ki, Kd)
                
                if self.objective == 'H':
                    beta_vae_loss = recon_loss + self.beta * total_kld
                elif self.objective == 'B':
                    ### tricks for C
                    C = torch.clamp(self.C_max/self.C_stop_iter*self.global_iter, self.C_start, self.C_max.data[0])
                    beta_vae_loss = recon_loss + self.gamma*(total_kld-C).abs()
                    
                ### RTD regularization
                if self.use_rtd and self.rtd_reg is not None:
                    if self.global_iter >= self.rtd_iter_start:
                        i, rtd_loss = self.rtd_reg.compute_reg(self.net, x, None)
                        beta_vae_loss += self.gamma_rtd * rtd_loss
                ###

                self.optim.zero_grad()
                beta_vae_loss.backward()
                self.optim.step()
                
                if self.viz_on and self.global_iter%self.gather_step == 0:
                    self.gather.insert(iter=self.global_iter,
                                       mu=mu.mean(0).data, var=logvar.exp().mean(0).data,
                                       recon_loss=recon_loss.data, total_kld=total_kld.data,
                                       dim_wise_kld=dim_wise_kld.data, mean_kld=mean_kld.data, beta=self.beta)
                
                if self.global_iter%20 == 0:
                    ## write log to file
                    if self.objective == 'B':
                        C = C.item()
                    fw_log.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} exp_kld:{:.3f} beta:{:.4f}\n'.format(
                                self.global_iter, recon_loss.item(), total_kld.item(), C, self.beta))
                    ## write KL to file
                    dim_kl = dim_wise_kld.data.cpu().numpy()
                    dim_kl = [str(k) for k in dim_kl]
                    fw_kl.write('total_kld:{0:.3f}\t'.format(total_kld.item()))
                    fw_kl.write('z_dim:' + ','.join(dim_kl) + '\n')
                    
                    if self.global_iter%500 == 0:
                        fw_log.flush()
                        fw_kl.flush()
                    
                if self.viz_on and self.global_iter % self.gather_step==0:
                    self.gather.insert(images=x.data)
                    self.gather.insert(images=F.sigmoid(x_recon).data)
                    self.viz_reconstruction()
                    self.viz_lines()
                    self.gather.flush()

                if (self.viz_on or self.save_output) and self.global_iter%20000==0:
                    self.viz_traverse()

                if self.global_iter % self.save_step == 0:
                    self.save_checkpoint('last')
                    pbar.write('Saved checkpoint(iter:{})'.format(self.global_iter))
                    
                if self.global_iter%50000 == 0:
                    self.save_checkpoint(str(self.global_iter))

                if self.global_iter >= self.max_iter:
                    out = True
                    break
                    
        pbar.write("[Training Finished]")
        pbar.close()
        fw_log.close()
        

    def viz_reconstruction(self):
        self.net_mode(train=False)
        x = self.gather.data['images'][0][:100]
        x = make_grid(x, normalize=True)
        x_recon = self.gather.data['images'][1][:100]
        x_recon = make_grid(x_recon, normalize=True)
        images = torch.stack([x, x_recon], dim=0).cpu()
        self.viz.images(images, env=self.viz_name+'_reconstruction',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        if self.save_output:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            os.makedirs(output_dir, exist_ok=True)
            save_image(tensor=images, fp=os.path.join(output_dir, 'recon.jpg'), pad_value=1)
        self.net_mode(train=True)


    def viz_lines(self):
        self.net_mode(train=False)
        recon_losses = torch.stack(self.gather.data['recon_loss']).cpu()
        betas = torch.Tensor(self.gather.data['beta'])

        # mus = torch.stack(self.gather.data['mu']).cpu()
        # vars = torch.stack(self.gather.data['var']).cpu()
        
        dim_wise_klds = torch.stack(self.gather.data['dim_wise_kld'])
        # mean_klds = torch.stack(self.gather.data['mean_kld'])
        total_klds = torch.stack(self.gather.data['total_kld'])
        klds = torch.cat([dim_wise_klds, total_klds], 1).cpu()
        iters = torch.Tensor(self.gather.data['iter'])

        ## legend
        legend = []
        for z_j in range(self.z_dim):
            legend.append('z_{}'.format(z_j))
        # legend.append('mean')
        legend.append('total')

        if self.win_recon is None:
            self.win_recon = self.viz.line(
                                        X=iters,
                                        Y=recon_losses,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='reconsturction loss',))
        else:
            self.win_recon = self.viz.line(
                                        X=iters,
                                        Y=recon_losses,
                                        env=self.viz_name+'_lines',
                                        win=self.win_recon,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='reconsturction loss',))
        
        if self.win_beta is None:
            self.win_beta = self.viz.line(
                                        X=iters,
                                        Y=betas,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='beta',))
        else:
            self.win_beta = self.viz.line(
                                        X=iters,
                                        Y=betas,
                                        env=self.viz_name+'_lines',
                                        win=self.win_beta,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='beta',))

        if self.win_kld is None:
            self.win_kld = self.viz.line(
                                        X=iters,
                                        Y=klds,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend,
                                            xlabel='iteration',
                                            title='kl divergence',))
        else:
            self.win_kld = self.viz.line(
                                        X=iters,
                                        Y=klds,
                                        env=self.viz_name+'_lines',
                                        win=self.win_kld,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend,
                                            xlabel='iteration',
                                            title='kl divergence',))

        # if self.win_mu is None:
        #     self.win_mu = self.viz.line(
        #                                 X=iters,
        #                                 Y=mus,
        #                                 env=self.viz_name+'_lines',
        #                                 opts=dict(
        #                                     width=400,
        #                                     height=400,
        #                                     legend=legend[:self.z_dim],
        #                                     xlabel='iteration',
        #                                     title='posterior mean',))
        # else:
        #     self.win_mu = self.viz.line(
        #                                 X=iters,
        #                                 Y=vars,
        #                                 env=self.viz_name+'_lines',
        #                                 win=self.win_mu,
        #                                 update='append',
        #                                 opts=dict(
        #                                     width=400,
        #                                     height=400,
        #                                     legend=legend[:self.z_dim],
        #                                     xlabel='iteration',
        #                                     title='posterior mean',))

        # if self.win_var is None:
        #     self.win_var = self.viz.line(
        #                                 X=iters,
        #                                 Y=vars,
        #                                 env=self.viz_name+'_lines',
        #                                 opts=dict(
        #                                     width=400,
        #                                     height=400,
        #                                     legend=legend[:self.z_dim],
        #                                     xlabel='iteration',
        #                                     title='posterior variance',))
        # else:
        #     self.win_var = self.viz.line(
        #                                 X=iters,
        #                                 Y=vars,
        #                                 env=self.viz_name+'_lines',
        #                                 win=self.win_var,
        #                                 update='append',
        #                                 opts=dict(
        #                                     width=400,
        #                                     height=400,
        #                                     legend=legend[:self.z_dim],
        #                                     xlabel='iteration',
        #                                     title='posterior variance',))
        self.net_mode(train=True)

    def viz_traverse(self, limit=3, inter=2/3, loc=-1):
        self.net_mode(train=False)
        import random

        decoder = self.net.decoder
        encoder = self.net.encoder
        interpolation = torch.arange(-limit, limit+0.1, inter)
        
        n_dsets = len(self.data_loader.dataset)
        rand_idx = random.randint(1, n_dsets-1)

        random_img = self.data_loader.dataset.__getitem__(rand_idx)
        random_img = Variable(cuda(random_img, self.use_cuda), volatile=True).unsqueeze(0)
        random_img_z = encoder(random_img)[:, :self.z_dim]

        random_z = Variable(cuda(torch.rand(1, self.z_dim), self.use_cuda), volatile=True)

        if self.dataset == 'dsprites':
            fixed_idx1 = 87040 # square
            fixed_idx2 = 332800 # ellipse
            fixed_idx3 = 578560 # heart

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)
            fixed_img1 = Variable(cuda(fixed_img1, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)
            fixed_img2 = Variable(cuda(fixed_img2, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)
            fixed_img3 = Variable(cuda(fixed_img3, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]
            
            Z = {'fixed_square':fixed_img_z1, 'fixed_ellipse':fixed_img_z2,
                 'fixed_heart':fixed_img_z3, 'random_img':random_img_z}
        else:
            fixed_idx = 0
            fixed_img = self.data_loader.dataset.__getitem__(fixed_idx)
            fixed_img = Variable(cuda(fixed_img, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z = encoder(fixed_img)[:, :self.z_dim]
            Z = {'fixed_img':fixed_img_z, 'random_img':random_img_z, 'random_z':random_z}
            
        gifs = []
        for key in Z.keys():
            z_ori = Z[key]
            samples = []
            for row in range(self.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:
                    z[:, row] = val  ## row is the z latent variable
                    sample = F.sigmoid(decoder(z)).data
                    samples.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples, dim=0).cpu()
            title = '{}_latent_traversal(iter:{})'.format(key, self.global_iter)
            
            if self.viz_on:
                self.viz.images(samples, env=self.viz_name+'_traverse',
                                opts=dict(title=title), nrow=len(interpolation))

        if self.save_output:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            os.makedirs(output_dir, exist_ok=True)
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(Z), self.z_dim, len(interpolation), self.nc, 64, 64).transpose(1, 2)
            for i, key in enumerate(Z.keys()):
                for j, val in enumerate(interpolation):
                    save_image(tensor=gifs[i][j].cpu(),
                               fp=os.path.join(output_dir, '{}_{}.jpg'.format(key, j)),
                               nrow=self.z_dim, pad_value=1)

                grid2gif(os.path.join(output_dir, key+'*.jpg'),
                         os.path.join(output_dir, key+'.gif'), delay=10)

        self.net_mode(train=True)

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()

    def save_checkpoint(self, filename, silent=True):
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        win_states = {'recon':self.win_recon,
                      'beta': self.win_beta,
                      'kld':self.win_kld,
                    #   'mu':self.win_mu,
                    #   'var':self.win_var,
                      }
        states = {'iter':self.global_iter,
                  'win_states':win_states,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))


    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.win_recon = checkpoint['win_states']['recon']
            self.win_kld = checkpoint['win_states']['kld']
            # self.win_var = checkpoint['win_states']['var']
            # self.win_mu = checkpoint['win_states']['mu']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
        
