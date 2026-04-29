import torch
from poincare import PoincareManifold


class RiemanianSGD(torch.optim.Optimizer):
    def __init__(self, params, lr, manifold=None, **kwargs):
        if manifold is None:
            manifold = PoincareManifold()
        defaults = dict(lr=lr, manifold=manifold)
        super(RiemanianSGD, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, lr=None, counts=None, closure=None):
        '''
        Une étape de descente de gradient Riemannienne : à savoir calcul de proj(theta+lr*rgrad)
        '''
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = lr if lr is not None else group["lr"]
            manifold = group["manifold"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if d_p.is_sparse:
                    d_p = d_p.coalesce()
                d_p = manifold.rgrad(p.data, d_p)
                manifold.euclidean_retractation(p, d_p, lr)
        return loss