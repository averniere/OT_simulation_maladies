import torch
import numpy as np
from torch.autograd import Function


def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()


def artanh(x):
    return Artanh.apply(x)


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-15, 1 - 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (torch.log_(1 + z).sub_(torch.log_(1 - z))).mul_(0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


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
    def __init__(self, eps=1e-2, K=None, max_norm=1, **kwargs):
        self.eps = eps
        self.max_norm = max_norm-eps
        self.min_norm = 1e-15
        self.K = K
        if K is not None:
            self.inner_radius = 2 * K / (1 + np.sqrt(1 + 4 * K * self.K))

    def normalize(self, u, c):
        return self.proj_x(u,c)

    def proj_x(self, x, c):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps) / (c ** 0.5)
        scale = (maxnorm / norm).clamp(max=1.0)
        return x * scale    

    def distance(self, u, v, c):
        sqrt_c = c ** 0.5
        mob = self.mobius_add(-u, v, c, dim=-1)
        mob_norm = mob.norm(dim=-1, p=2).clamp(1e-10, 1. - 1e-5)
        return 2 / sqrt_c * artanh(sqrt_c * mob_norm)

    def mobius_add(self, x, y, c, dim=-1):
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        return num / denom.clamp_min(self.min_norm)
    
    def lambda_x(self, x, c):
        x_sqnorm = torch.sum(x.pow(2), dim=-1, keepdim=True)
        return 2 / (1. - c * x_sqnorm).clamp_min(self.min_norm)

    def egrad2rgrad(self, p, dp, c):
        """Remplace rgrad — version correcte avec courbure variable."""
        lambda_p = self.lambda_x(p, c)
        dp = dp / lambda_p.pow(2)
        return dp

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
