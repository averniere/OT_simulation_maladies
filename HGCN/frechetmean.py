import torch
import torch.nn as nn


TOLEPS = {torch.float32: 1e-6, torch.float64: 1e-12}
EPS = {torch.float32: 1e-4, torch.float64: 1e-8}


class Acosh(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        x = torch.clamp(x, min=1+EPS[x.dtype])
        z = torch.sqrt(x * x - 1)
        ctx.save_for_backward(z)
        return torch.log(x + z)

    @staticmethod
    def backward(ctx, g):
        z, = ctx.saved_tensors
        z.data.clamp(min=EPS[z.dtype])
        z = g / z
        return z, None


arcosh = Acosh.apply


def darcosh(x):
    cond = (x < 1 + 1e-7)
    x = torch.where(cond, 2 * torch.ones_like(x), x)
    x = torch.where(~cond, 2 * arcosh(x) / torch.sqrt(x**2 - 1), x)
    return x


def d2arcosh(x):
    cond = (x < 1 + 1e-7)
    x = torch.where(cond, -2/3 * torch.ones_like(x), x)
    x = torch.where(~cond, 2 / (x**2 - 1) - 2 * x * arcosh(x) / ((x**2 - 1)**(3/2)), x)
    return x


def l_prime(y):
    cond = y < 1e-12
    val = 4 * torch.ones_like(y)
    ret = torch.where(cond, val, 2 * arcosh(1 + 2 * y) / (y.pow(2) + y).sqrt())
    return ret


def frechet_ball_forward(X, w, K=-1.0, max_iter=1000, rtol=1e-6, atol=1e-6, verbose=False):
    """
    Args
    ----
        X (tensor): point of shape [..., points, dim]
        w (tensor): weights of shape [..., points]
        K (float): curvature (must be negative)
    Returns
    -------
        frechet mean (tensor): shape [..., dim]
    """
    mu = X[..., 0, :].clone()

    x_ss = X.pow(2).sum(dim=-1)

    mu_prev = mu
    iters = 0
    for _ in range(max_iter):
        mu_ss = mu.pow(2).sum(dim=-1)
        xmu_ss = (X - mu.unsqueeze(-2)).pow(2).sum(dim=-1)

        alphas = l_prime(-K * xmu_ss / ((1 + K * x_ss) * (1 + K * mu_ss.unsqueeze(-1)))) / (1 + K * x_ss)

        alphas = alphas * w

        c = (alphas * x_ss).sum(dim=-1)
        b = (alphas.unsqueeze(-1) * X).sum(dim=-2)
        a = alphas.sum(dim=-1)

        b_ss = b.pow(2).sum(dim=-1)

        eta = (a - K * c - ((a - K * c).pow(2) + 4 * K * b_ss).sqrt()) / (2 * (-K) * b_ss)

        mu = eta.unsqueeze(-1) * b

        dist = (mu - mu_prev).norm(dim=-1)
        prev_dist = mu_prev.norm(dim=-1)
        if (dist < atol).all() or (dist / prev_dist < rtol).all():
            break

        mu_prev = mu
        iters += 1

    if verbose:
        print(iters)

    return mu


def grad_var(X, y, w, K):
    """
    Args
    ----
        X (tensor): point of shape [..., points, dim]
        y (tensor): mean point of shape [..., dim]
        w (tensor): weight tensor of shape [..., points]
        K (float): curvature (must be negative)

    Returns
    -------
        grad (tensor): gradient of variance [..., dim]
    """
    yl = y.unsqueeze(-2)
    xnorm = 1 + K * X.norm(dim=-1).pow(2)
    ynorm = 1 + K * yl.norm(dim=-1).pow(2)
    xynorm = (X - yl).norm(dim=-1).pow(2)

    D = xnorm * ynorm
    v = 1 - 2 * K * xynorm / D

    Dl = D.unsqueeze(-1)
    vl = v.unsqueeze(-1)

    first_term = (X - yl) / Dl
    sec_term = K / Dl.pow(2) * yl * xynorm.unsqueeze(-1) * xnorm.unsqueeze(-1)
    return -(4 * darcosh(vl) * w.unsqueeze(-1) * (first_term + sec_term)).sum(dim=-2)
    

def inverse_hessian(X, y, w, K):
    """
    Args
    ----
        X (tensor): point of shape [..., points, dim]
        y (tensor): mean point of shape [..., dim]
        w (tensor): weight tensor of shape [..., points]
        K (float): curvature (must be negative)

    Returns
    -------
        inv_hess (tensor): inverse hessian of [..., points, dim, dim]
    """
    yl = y.unsqueeze(-2)
    xnorm = 1 + K * X.norm(dim=-1).pow(2)
    ynorm = 1 + K * yl.norm(dim=-1).pow(2)
    xynorm = (X - yl).norm(dim=-1).pow(2)

    D = xnorm * ynorm
    v = 1 - 2 * K * xynorm / D

    Dl = D.unsqueeze(-1)
    vl = v.unsqueeze(-1)
    vll = vl.unsqueeze(-1)

    """
    \partial T/ \partial y
    """
    first_const = -8 * (K ** 2) * xnorm / D.pow(2)
    matrix_val = (first_const.unsqueeze(-1) * yl).unsqueeze(-1) * (X - yl).unsqueeze(-2)
    first_term = matrix_val + matrix_val.transpose(-1, -2)

    sec_const = -16 * (K ** 3) * xnorm.pow(2) / D.pow(3) * xynorm
    sec_term = (sec_const.unsqueeze(-1) * yl).unsqueeze(-1) * yl.unsqueeze(-2)

    third_const = -4 * K / D + 4 * (K ** 2) * xnorm /D.pow(2) * xynorm
    third_term = third_const.reshape(*third_const.shape, 1, 1) * torch.eye(y.shape[-1]).to(X).reshape((1, ) * len(third_const.shape) + (y.shape[-1], y.shape[-1]))

    Ty = first_term + sec_term + third_term

    """
    T
    """
    
    first_term = K / Dl * (X - yl)
    sec_term = K.pow(2) / Dl.pow(2) * yl * xynorm.unsqueeze(-1) * xnorm.unsqueeze(-1)
    T = 4 * (first_term + sec_term)

    """
    inverse of shape [..., points, dim, dim]
    """
    first_term = d2arcosh(vll) * T.unsqueeze(-1) * T.unsqueeze(-2)
    sec_term = darcosh(vll) * Ty
    hessian = ((first_term + sec_term) * w.unsqueeze(-1).unsqueeze(-1)).sum(dim=-3) / -K
    inv_hess = torch.inverse(hessian)
    return inv_hess


def frechet_ball_backward(X, y, grad, w, K):
    """
    Args
    ----
        X (tensor): point of shape [..., points, dim]
        y (tensor): mean point of shape [..., dim]
        grad (tensor): gradient
        K (float): curvature (must be negative)

    Returns
    -------
        gradients (tensor, tensor, tensor): 
            gradient of X [..., points, dim], weights [..., dim], curvature []
    """
    if not torch.is_tensor(K):
        K = torch.tensor(K).to(X)

    with torch.no_grad():
        inv_hess = inverse_hessian(X, y, w=w, K=K)

    with torch.enable_grad():
        # clone variables
        X = nn.Parameter(X.detach())
        y = y.detach()
        w = nn.Parameter(w.detach())
        K = nn.Parameter(K)

        grad = (inv_hess @ grad.unsqueeze(-1)).squeeze()
        gradf = grad_var(X, y, w, K)
        dx, dw, dK = torch.autograd.grad(-gradf, (X, w, K), grad)

    return dx, dw, dK


class FrechetMean(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, w, K):
        mean = frechet_ball_forward(x, w, K, rtol=TOLEPS[x.dtype], atol=TOLEPS[x.dtype])
        ctx.save_for_backward(x, mean, w, K)
        return mean

    @staticmethod
    def backward(ctx, grad_output):
        X, mean, w, K, manifold_id = ctx.saved_tensors
        dx, dw, dK = frechet_ball_backward(X, mean, grad_output, w, K)
        return dx, dw, dK, None


def frechet_mean(x, K, w=None):
    if w is None:
        w = torch.ones(x.shape[:-1]).to(x)
    return FrechetMean.apply(x, w, K)