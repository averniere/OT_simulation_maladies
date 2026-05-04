import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

import encoder
from poincare import PoincareManifold
from decoder import FermiDiracDecoder


class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args, device):
        super(BaseModel, self).__init__()
        if args.c is not None:
            self.c = torch.tensor([args.c])
            self.c = self.c.to(device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = PoincareManifold()
        #if self.manifold.name == 'Hyperboloid':
            #args.feat_dim = args.feat_dim + 1
        self.nnodes = args.n_nodes
        self.encoder = encoder.HGCN(self.c, args, device)
        self.input_embeddings = nn.Embedding(self.nnodes, args.feat_dim)
        nn.init.uniform_(self.input_embeddings.weight, -1e-5, 1e-5)

    def encode(self, x, adj):
        #if self.manifold.name == 'Hyperboloid':
            #o = torch.zeros_like(x)
            #x = torch.cat([o[:, 0:1], x], dim=1)
        #x = self.input_embeddings(torch.arange(self.nnodes).to(x.device))
        return self.encoder.encode(x, adj)

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class LPModel(BaseModel):
    """
    Base model for link prediction task.
    """

    def __init__(self, args, device):
        super(LPModel, self).__init__(args, device)
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges

    def decode(self, h, idx):
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        def check_grad(name):
            def hook(grad):
                if torch.isnan(grad).any():
                    print(f"NaN dans grad de {name}")
                else:
                    print(f"grad de {name} OK: min={grad.min().item():.6f}, max={grad.max().item():.6f}")
            return hook
    
        # emb_in.register_hook(check_grad("emb_in"))
        # emb_out.register_hook(check_grad("emb_out"))
        
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        probs = self.dc.forward(sqdist)
        return probs

    def compute_metrics(self, embeddings, data, split):
        if split == 'train':
            edges_false = data[f'{split}_edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)]
        else:
            edges_false = data[f'{split}_edges_false']
        # print("COMPUTE METRICS")
        # print("indices neg max:", edges_false.max(), "nb embeddings:", embeddings.shape[0])
        # print("embeddings NaN ?", torch.isnan(embeddings).any())
        # print("embeddings aux indices neg NaN ?", torch.isnan(embeddings[edges_false[:, 0]]).any(), torch.isnan(embeddings[edges_false[:, 1]]).any())
        pos_scores = self.decode(embeddings, data[f'{split}_edges'])
        neg_scores = self.decode(embeddings, edges_false)
        # print("pos_scores avant clamp NaN ?", torch.isnan(pos_scores).any())
        # print("pos_scores avant clamp min/max :", pos_scores.min().item(), pos_scores.max().item())
        
        pos_scores = pos_scores.clamp(1e-7, 1 - 1e-7)
        neg_scores = neg_scores.clamp(1e-7, 1 - 1e-7)

        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))

        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()

        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        return metrics

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])
        