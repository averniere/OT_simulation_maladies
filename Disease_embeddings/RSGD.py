import torch
from poincare import PoincareManifold

class RiemanianSGD(torch.optim.Optimizer):
    def __init__(self, params, lr, manifold=None, c=1.0, **kwargs):
        if manifold is None:
            manifold = PoincareManifold()
        defaults = dict(lr=lr, manifold=manifold, c=c)
        super(RiemanianSGD, self).__init__(params, defaults)

    def update_c(self, c):
        """Permet de mettre à jour c depuis l'extérieur (utile si c est appris)."""
        for group in self.param_groups:
            group['c'] = c

    @torch.no_grad()
    def step(self, lr=None, counts=None, closure=None):
        '''
        Une étape de descente de gradient Riemannienne : à savoir calcul de proj(theta+lr*rgrad)
        '''
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            effective_lr = lr if lr is not None else group["lr"]
            manifold = group["manifold"]
            c = group["c"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if d_p.is_sparse:
                    d_p = d_p.coalesce()
                d_p = manifold.egrad2rgrad(p.data, d_p, c)
                p.data = p.data = manifold.normalize(p.data + (-effective_lr) * d_p, c)
        return loss
        