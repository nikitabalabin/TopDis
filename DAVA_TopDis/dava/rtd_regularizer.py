from rtd import *
from scipy.special import softmax
import torch

normal_s = lambda x: 0.5 * (torch.erf(x/np.sqrt(2)) + 1)
normal_sinv = lambda x: np.sqrt(2) * torch.erfinv(2 * x - 1)

class RTDRegularizer:
    def __init__(self, lp, q_normalize, sample_based, weightnorm_sampler):
        self.rtd = MinMaxRTDLoss(dim=1, lp=lp,  **{"engine":"ripser", "is_sym":True, "card":50})
        self.q_normalize = q_normalize
        self.sample_based = sample_based
        self.weightnorm_sampler = weightnorm_sampler
        
    def compute_reg(self, model, x_batch, current_kl_step):
        if self.sample_based:
            _, z, _ = model(x_batch, current_kl_step)
        else:
            z, _ = model.encoder(x_batch)
        if self.weightnorm_sampler:
            importance = model.decoder.decode[0].weight.norm(dim=0).flatten().detach().clone().cpu()
            probs = importance.numpy()
            probs = probs / max(probs.sum(), 1e-6)
            i = np.random.choice(model.z_dim, p=probs)
        else:
            i = np.random.choice(model.z_dim)
        m_batch = z[:,i].mean(0, keepdim=True)
        s_batch = z[:,i].std(0, keepdim=True)
        z_norm = (z[:,i] - m_batch) / s_batch
        prob = normal_s(z_norm)
        C = 1/8
        sgn = torch.sign(torch.randn(1)).item()
        if sgn > 0:
            mask = (prob + C < 1)
        else:
            mask = (prob - C > 0)
            C = -C
        z_valid = z[mask].clone()
        z_new = z_valid.clone()
        z_new[:, i] = normal_sinv(prob[mask] + C) * s_batch + m_batch
        z_reg = torch.cat([z_valid, z_new])
        x_reg = model.decode(z_reg)
        cloud1, cloud2 = x_reg.chunk(2)
        cloud1, q11, q12 = z_dist(cloud1.flatten(1,-1), q_normalize=self.q_normalize)
        cloud2, q21, q22 = z_dist(cloud2.flatten(1,-1), q_normalize=self.q_normalize)
        *_, rtd_pos = self.rtd(cloud1, cloud2)
        return i, rtd_pos