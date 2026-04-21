import numpy as np
import random
import torch
from collections import deque
from numpy.random import default_rng


def compute_depths(edge_index, num_nodes):
    """
    Calcule la profondeur de chaque nœud via BFS depuis les racines.
    edge_index : (2, E) — (enfants-parent)
    """
    children = set(edge_index[0].tolist())
    roots = [n for n in range(num_nodes) if n not in children]

    depth = torch.full((num_nodes,), -1, dtype=torch.long)
    queue = deque()

    for r in roots:
        depth[r] = 0
        queue.append(r)

    # Construire liste d'adjacence parent → enfants
    adj_list = {i: [] for i in range(num_nodes)}
    for child, parent in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        adj_list[parent].append(child)

    while queue:
        node = queue.popleft()
        for child in adj_list[node]:
            if depth[child] == -1:
                depth[child] = depth[node] + 1
                queue.append(child)

    max_depth = depth[depth >= 0].max().item()
    depth[depth == -1] = max_depth

    return depth

class BatchedDataset:
    """
    - Paires positives  : arêtes (u, v) du graphe.
    - Paires négatives  : noeuds non voisins de u tirés uniforméments.
    - Chaque batch est un LongTensor (B, 2+nnegs) :
        col 0 = u
        col 1 = v voisins
        col 2..end  = non voisins (négatifs)
    """
 
    def __init__(self, edges, objects, nnegs, batch_size, burnin=False, depth_temperature = 1.0):
        self.edges = edges
        self.N = len(objects)
        self.nnegs = nnegs
        self.batch_size = batch_size
        self.burnin = burnin
        self.depth_temperature = depth_temperature
        self.rng = default_rng()
 
        # Voisins
        self.pos_neighbors = [set() for _ in range(self.N)]
        counts = np.zeros(self.N, dtype=np.float64)
        for u, v in edges:
            self.pos_neighbors[int(u)].add(int(v))
            counts[int(v)] += 1.0    
        
        # Profondeurs
        edge_index = torch.tensor(edges, dtype=torch.long).T
        self.depths = compute_depths(edge_index, self.N).numpy()
        self._build_alias_tables()


    def _build_alias_tables(self):
        """Calcule pour chaque profondeur unique un vecteur de pondération/proba sur les N noeuds."""
        self.alias_tables = {}
        for d in np.unique(self.depths):
            diff = np.abs(self.depths - d).astype(np.float32)
            w = np.exp(-diff / self.depth_temperature)
            w /= w.sum()
            self.alias_tables[d] = self._make_alias(w)

    @staticmethod
    def _make_alias(probs):
        """Construit la alias table (J, q) pour un tirage O(1)."""
        N = len(probs)
        q = np.zeros(N, dtype=np.float32)
        J = np.zeros(N, dtype=np.int64)
        smaller, larger = [], []

        for i, p in enumerate(probs * N):
            q[i] = p
            (smaller if p < 1.0 else larger).append(i)

        while smaller and larger:
            s, l = smaller.pop(), larger.pop()
            J[s] = l
            q[l] = q[l] + q[s] - 1.0
            (smaller if q[l] < 1.0 else larger).append(l)

        return J, q


    def _alias_draw(self, J, q, size):
        """Tirage vectorisé O(size) grâce à la alias table."""
        k = self.rng.integers(0, self.N, size=size)
        r = self.rng.random(size=size)
        return np.where(r < q[k], k, J[k])


    def _sample_neg(self, u, target_depth, nnegs):

        J, q = self.alias_tables[target_depth]
        candidates = self._alias_draw(J, q, size=nnegs * 2)
        pos = self.pos_neighbors[u]
        invalid = np.array([c == u or c in pos for c in candidates])
        valid = candidates[~invalid][:nnegs]

        n_missing = nnegs - len(valid)
        if n_missing > 0:
            fallback = self.rng.integers(0, self.N, size=n_missing)
            valid = np.concatenate([valid, fallback])

        return valid


    def __iter__(self):
        perm = np.random.permutation(len(self.edges))
        for start in range(0, len(self.edges), self.batch_size):
            chunk = perm[start:start + self.batch_size]
            B = len(chunk)
            ix = np.empty((B, 2 + self.nnegs), dtype=np.int64)
            for j, idx in enumerate(chunk):
                u, v  = self.edges[idx]
                ix[j, 0] = u
                ix[j, 1] = v
                target_depth = self.depths[int(v)]
                ix[j, 2:] = self._sample_neg(int(u), target_depth, self.nnegs)
            yield torch.from_numpy(ix)
 
    def __len__(self):
        return int(np.ceil(len(self.edges) / self.batch_size))