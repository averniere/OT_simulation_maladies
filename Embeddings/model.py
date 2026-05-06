import torch
import torch.nn as nn
import torch.nn.functional as F



class Poincarre_embeddings(nn.Module):

    def __init__(self, n, dim, manifold, sparse, learn_curvature, init_curvature, weight_decay=1e-4):
        super(Poincarre_embeddings, self).__init__()
        self.embeddings = nn.Embedding(n, dim, sparse=sparse)
        self.manifold = manifold
        self.n = n
        self.weight_decay = weight_decay
        nn.init.uniform_(self.embeddings.weight, -1e-3, 1e-3)
        if learn_curvature:
            self._log_c = nn.Parameter(torch.tensor(float(init_curvature)).log())
        else:
            self.register_buffer('_log_c', torch.tensor(float(init_curvature)).log())

    @property
    def c(self):
        """Courbure toujours positive via exp."""
        return self._log_c.exp()

    @property
    def weight(self):
        return self.embeddings.weight

    def forward(self, inputs):
        return self.embeddings(inputs)

    def regularization_loss(self):
        if self.weight_decay == 0.0:
            return 0.0
        norms = self.embeddings.weight.norm(dim=-1)
        max_norm = 1.0 / self.c.sqrt()
        penalty = -torch.log(max_norm - norms + self.manifold.eps).mean()
        return self.weight_decay * penalty
    

class Distance_PE(Poincarre_embeddings):
    def energy(self, s, o):
        return self.manifold.distance(s, o, self.c)

    def loss(self, inp, target, **kwargs):
        ce = F.cross_entropy(inp.neg(), target)
        reg = self.regularization_loss()
        return ce + reg
        #return F.cross_entropy(inp.neg(), target)
 

    



    