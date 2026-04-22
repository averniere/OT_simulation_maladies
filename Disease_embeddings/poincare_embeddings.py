import torch
import torch.nn as nn
import torch.nn.functional as F


def klSym(preds, targets):
    # preds = preds + eps
    # targets = targets + eps
    logPreds = preds.clamp(1e-20).log()
    logTargets = targets.clamp(1e-20).log()
    diff = targets - preds
    return (logTargets * diff - logPreds * diff).sum() / len(preds)


class Poincarre_embeddings(nn.Module):

    def __init__(self, n, dim, manifold, sparse, Qdist='Laplace', lossfn='klSym', gamma=1.0, cuda=0):
        super(Poincarre_embeddings, self).__init__()
        self.dim = dim
        self.n = n
        self.manifold = manifold
        self.embeddings = nn.Embedding(n, dim, max_norm=manifold.max_norm)
        self.Qdist = Qdist
        self.lossfnname = lossfn
        self.gamma = gamma
        nn.init.uniform_(self.embeddings.weight, -1e-3, 1e-3)

        self.sm = nn.Softmax(dim=1)
        self.lsm = nn.LogSoftmax(dim=1)

        if lossfn == 'kl':
            self.lossfn = nn.KLDivLoss()
        elif lossfn == 'klSym':
            self.lossfn = klSym
        elif lossfn == 'mse':
            self.lossfn = nn.MSELoss()
        else:
            raise NotImplementedError

        if cuda:
            self.lt.cuda()


    def forward(self, inputs):
        embs_all = self.embeddings.weight.unsqueeze(0)
        embs_all = embs_all.expand(len(inputs), self.size, self.dim)

        embs_inputs = self.embeddings(inputs).unsqueeze(1)
        embs_inputs = embs_inputs.expand_as(embs_all)

        dists = self.manifold.distance.apply(embs_inputs, embs_all).squeeze(-1)

        if self.lossfnname == 'kl':
            if self.Qdist == 'laplace':
                return self.lsm(-self.gamma * dists)
            elif self.Qdist == 'gaussian':
                return self.lsm(-self.gamma * dists.pow(2))
            elif self.Qdist == 'student':
                return self.lsm(-torch.log(1 + self.gamma * dists))
            else:
                raise NotImplementedError

        elif self.lossfnname == 'klSym':
            if self.Qdist == 'laplace':
                return self.sm(-self.gamma * dists)
            elif self.Qdist == 'gaussian':
                return self.sm(-self.gamma * dists.pow(2))
            elif self.Qdist == 'student':
                return self.sm(-torch.log(1 + self.gamma * dists))
            else:
                raise NotImplementedError

        elif self.lossfnname == 'mse':
            return self.sm(-self.gamma * dists)
            
        else:
            raise NotImplementedError

    @property
    def weight(self):
        return self.embeddings.weight
