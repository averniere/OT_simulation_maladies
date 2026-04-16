import torch
import numpy as np
from torch.autograd import Function


class Distance(Function):
    '''
    Distance sur la boule de Poincaré et son gradient.
    '''

    @staticmethod
    def grad(x, v, sqnormx, sqnormv, sqdist, eps):
        '''
        Dérivée partielle de la distance de Poincaré par rapport à x.
        '''
        alpha = (1 - sqnormx)
        beta = (1 - sqnormv)
        gamma = 1 + 2 * sqdist / (alpha * beta)
        a = ((sqnormv - 2 * torch.sum(x * v, dim=-1) + 1) / torch.pow(alpha, 2))\
            .unsqueeze(-1).expand_as(x)
        a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
        gamma = torch.sqrt(torch.pow(gamma, 2) - 1)
        gamma = torch.clamp(gamma * beta, min=eps).unsqueeze(-1)
        return 4 * a / gamma.expand_as(a)

    @staticmethod
    def forward(ctx, u, v, eps):
        squnorm = torch.clamp(torch.sum(u * u, dim=-1), 0, 1 - eps)
        sqvnorm = torch.clamp(torch.sum(v * v, dim=-1), 0, 1 - eps)
        sqdist = torch.sum(torch.pow(u - v, 2), dim=-1)
        ctx.eps = eps
        ctx.save_for_backward(u, v, squnorm, sqvnorm, sqdist)
        x = sqdist / ((1 - squnorm) * (1 - sqvnorm)) * 2 + 1
        # arcosh
        z = torch.sqrt(torch.pow(x, 2) - 1)
        return torch.log(x + z)  # cf Wikipédia : arcosh(x)=ln(x+sqrt(x2-1))

    @staticmethod
    def backward(ctx, g):
        u, v, squnorm, sqvnorm, sqdist = ctx.saved_tensors
        g = g.unsqueeze(-1)
        gu = Distance.grad(u, v, squnorm, sqvnorm, sqdist, ctx.eps)
        gv = Distance.grad(v, u, sqvnorm, squnorm, sqdist, ctx.eps)
        return g.expand_as(gu) * gu, g.expand_as(gv) * gv, None



class PoincareManifold():
    def __init__(self, eps=1e-5, K=None, max_norm=1, **kwargs):
        self.eps = eps
        self.max_norm = max_norm-eps
        self.K = K
        if K is not None:
            self.inner_radius = 2 * K / (1 + np.sqrt(1 + 4 * K * self.K))


    def normalize(self, u):
        d = u.size(-1)
        if self.max_norm:
            u.view(-1, d).renorm_(2, 0, self.max_norm) # ou 2 ?
        return u


    def distance(self, u, v):
        return Distance.apply(u, v, self.eps)


    def half_aperture(self, u):
        eps = self.eps
        sqnu = u.pow(2).sum(dim=-1)
        sqnu.clamp_(min=0, max=1 - eps)
        return torch.asin((self.inner_radius * (1 - sqnu) / torch.sqrt(sqnu))
            .clamp(min=-1 + eps, max=1 - eps))


    def angle_at_u(self, u, v):
        norm_u = u.norm(2, dim=-1)
        norm_v = v.norm(2, dim=-1)
        dot_prod = (u * v).sum(dim=-1)
        edist = (u - v).norm(2, dim=-1)  # euclidean distance
        num = (dot_prod * (1 + norm_v ** 2) - norm_v ** 2 * (1 + norm_u ** 2))
        denom = (norm_v * edist * (1 + norm_v**2 * norm_u**2 - 2 * dot_prod).sqrt())
        return (num / denom).clamp_(min=-1 + self.eps, max=1 - self.eps).acos()


    def rgrad(self, p, d_p):
        '''
        Calcule le gradient riemannien : d_p*(1-||p||^2)/4
        p : theta dans le papier (embeddings).
        d_p : gradient euclidien en p (de la fonction de perte dans le papier)
        '''
        if d_p.is_sparse:
            p_sqnorm = torch.sum(
                p[d_p._indices()[0].squeeze()] ** 2, dim=1,
                keepdim=True
            ).expand_as(d_p._values())
            n_vals = d_p._values() * ((1 - p_sqnorm) ** 2) / 4
            #n_vals.renorm_(2, 0, 5)
            d_p = torch.sparse_coo_tensor(d_p._indices(), n_vals, d_p.size())
        else:
            p_sqnorm = torch.sum(p ** 2, dim=-1, keepdim=True)
            d_p = d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(d_p)
        return d_p

