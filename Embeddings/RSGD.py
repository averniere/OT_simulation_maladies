import torch
from poincare import PoincareManifold

class RiemanianSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, manifold=None, **kwargs):
        if manifold is None:
            manifold = PoincareManifold()
        defaults = dict(lr=lr, manifold=manifold)
        super(RiemanianSGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            manifold = group["manifold"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if d_p.is_sparse:
                    d_p = d_p.coalesce()
                d_p = manifold.rgrad(p.data, d_p)
                d_p.mul_(-lr)
                d_p.add(p.data)
                p.data = manifold.normalize(d_p)
                # ou p = manifold.normalize(d_p)?
        return loss