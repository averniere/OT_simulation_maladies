import numpy as np
import random
import torch

class BatchedDataset:
    """
    - Paires positives  : arêtes (u, v) du graphe.
    - Paires négatives  : noeuds non voisins de u tirés uniforméments.
    - Chaque batch est un LongTensor (B, 2+nnegs) :
        col 0 = u
        col 1 = v voisins
        col 2..end  = non voisins (négatifs)
    """
 
    def __init__(self, edges, objects, nnegs, batch_size, burnin=False):
        self.edges = edges
        self.N = len(objects)
        self.nnegs = nnegs
        self.batch_size = batch_size
        self.burnin = burnin
 
        # Voisins
        self.pos_neighbors = [set() for _ in range(self.N)]
        counts = np.zeros(self.N, dtype=np.float64)
        for u, v in edges:
            self.pos_neighbors[int(u)].add(int(v))
            counts[int(v)] += 1.0
 
    
    def _sample_neg(self, u):
        pos = self.pos_neighbors[u]
        for _ in range(10 * self.nnegs):
            n = random.randint(0, self.N - 1)
            if n != u and n not in pos:
                return n
        return random.randint(0, self.N - 1) 
 
    def __iter__(self):
        perm = np.random.permutation(len(self.edges))
        for start in range(0, len(self.edges), self.batch_size):
            chunk = perm[start:start + self.batch_size]
            B     = len(chunk)
            ix    = np.empty((B, 2 + self.nnegs), dtype=np.int64)
            for j, idx in enumerate(chunk):
                u, v      = self.edges[idx]
                ix[j, 0]  = u
                ix[j, 1]  = v
                for k in range(self.nnegs):
                    ix[j, 2 + k] = self._sample_neg(int(u))
            yield torch.from_numpy(ix)
 
    def __len__(self):
        return int(np.ceil(len(self.edges) / self.batch_size))