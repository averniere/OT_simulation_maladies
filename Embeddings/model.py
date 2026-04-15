import torch
import torch.nn as nn
import torch.nn.functional as F



class Poincarre_embeddings(nn.Module):

    def __init__(self, n, dim, manifold, sparse):
        super(Poincarre_embeddings, self).__init__()
        self.embeddings = nn.Embedding(n, dim, sparse=sparse)
        self.manifold = manifold
        self.n = n
        nn.init.uniform_(self.embeddings.weight, -1e-3, 1e-3)

    def forward(self, inputs):
        return self.embeddings(inputs)
    
    
    @property
    def weight(self):
        return self.embeddings.weight
    

class Distance_PE(Poincarre_embeddings):
    def energy(self, s, o):
        return self.manifold.distance(s, o)

    def loss(self, inp, target, **kwargs):
        return F.cross_entropy(inp.neg(), target)
 

    



    