import numpy as np
import random
import torch
import scipy
import networkx as nx
import time
import numba
from tqdm import tqdm
from collections import deque
from scipy.sparse import csr_matrix
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
 
    def __init__(
        self, edges, objects, nnegs, batch_size, pos_neighbors, pos_ratio, burnin=False, 
        depth_temperature=1.0, max_edges_per_epoch=None
        ):
        self.edges = edges
        self.edges_arr = np.array(edges, dtype=np.int32)
        self.max_edges_per_epoch = max_edges_per_epoch
        self.N = len(objects)
        self.nnegs = nnegs
        self.pseudo_pos_ratio = pos_ratio
        self.batch_size = batch_size
        self.burnin = burnin
        self.depth_temperature = depth_temperature
        self.rng = default_rng()
 
        self.pos_neighbors = pos_neighbors
        self.pos_neighbors_arr = [np.array(list(s), dtype=np.int32) for s in self.pos_neighbors]
        rows = self.edges_arr[:, 0]
        cols = self.edges_arr[:, 1]
        self.pos_matrix = scipy.sparse.csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(self.N, self.N))
        
        # Profondeurs
        edge_index = torch.tensor(edges, dtype=torch.long).T
        self.depths = compute_depths(edge_index, self.N).numpy()
        self._build_alias_tables()


    def _build_hard_neg_candidates(self, edges, max_depth=2):
        G = nx.DiGraph()
        G.add_edges_from(edges)
    
        self.hard_neg_pool = {}
        for u in G.nodes():
            pos = self.pos_neighbors[u]
            # Nœuds à distance 2 ou 3 = proches structurellement mais pas voisins
            candidates = set()
            for depth in [2, 3]:
                for v in nx.single_source_shortest_path_length(G, u, cutoff=depth):
                    if v != u and v not in pos:
                        candidates.add(v)
            self.hard_neg_pool[u] = np.array(list(candidates))

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

    def _sample_neg(self, u, target_depth, nnegs, model=None, hard_ratio=0.5):

        J, q = self.alias_tables[target_depth]
        n_hard = int(nnegs * hard_ratio) if model is not None else 0
        n_easy = nnegs - n_hard
        pos = self.pos_neighbors[u]
        negs = []

        if n_hard > 0 and u in self.hard_neg_pool and len(self.hard_neg_pool[u]) > 0:
            pool = self.hard_neg_pool[u]
            chosen = self.rng.choice(pool, size=min(n_hard, len(pool)), replace=False)
            negs.append(chosen)
            n_easy += n_hard - len(chosen)
        if n_easy > 0:
            candidates = self._alias_draw(J, q, size=n_easy * 3)
            invalid = np.array([c == u or c in pos for c in candidates])
            valid = candidates[~invalid][:n_easy]
            n_missing = n_easy - len(valid)
            if n_missing > 0:
                fallback = self.rng.integers(0, self.N, size=n_missing)
                valid = np.concatenate([valid, fallback])
            negs.append(valid[:n_easy])

        return np.concatenate(negs)
    
    def _check_pos(self, us, candidates):
        B, K = candidates.shape
        rows = np.repeat(us, K)
        cols = candidates.ravel()
        mask = np.asarray(self.pos_matrix[rows, cols]).ravel().astype(bool)
        return mask.reshape(B, K)

    def __iter__(self, model=None, hard_ratio=0.0):
    
        perm = np.random.permutation(len(self.edges_arr))
    
        for start in range(0, len(self.edges_arr), self.batch_size):
            chunk = perm[start:start + self.batch_size]
            B = len(chunk)
            batch_edges = self.edges_arr[chunk]
            us = batch_edges[:, 0]
            vs = batch_edges[:, 1]

            # On échantillonne un pourcentage de noeuds pseudo positifs
            if self.pseudo_pos_ratio > 0 :
                n_pseudo = max(1, int(B * self.pseudo_pos_ratio))  # ex: ratio=0.2, B=100 → 20 paires
                pseudo_us, pseudo_vs = [], []
                edges_set = set(map(tuple, batch_edges))  # pour éviter les doublons du batch courant
                for _ in range(n_pseudo * 5):  # oversample puis filtre
                    if len(pseudo_us) >= n_pseudo:
                        break
                    # Tire un u aléatoire dans le batch
                    u = us[np.random.randint(B)]
                    pseudo_neighbors = self.pos_neighbors[int(u)]
                    if len(pseudo_neighbors) == 0:
                        continue
                    v = int(np.random.choice(list(pseudo_neighbors)))
                    if (u, v) not in edges_set:  # pas déjà dans les arêtes graphe
                        pseudo_us.append(u)
                        pseudo_vs.append(v)
                        edges_set.add((u, v))
                if pseudo_us:
                    us = np.concatenate([us, pseudo_us])
                    vs = np.concatenate([vs, pseudo_vs])
                    B = len(us)
            
            ix = np.empty((B, 2 + self.nnegs), dtype=np.int64)
            ix[:, 0] = us
            ix[:, 1] = vs

            target_depths = self.depths[vs]
            #pos_lists = [self.pos_neighbors[u] for u in us]  # Voisins positifs des u du batch
            pos_lists = None

            for d in np.unique(target_depths):
                mask = target_depths == d
                n = mask.sum()
                J, q = self.alias_tables[d]
                batch_us = us[mask]

                candidates = self._alias_draw(J, q, size=n * self.nnegs * 4).reshape(n, -1)

                is_self = candidates == batch_us[:, None]

                invalid = is_self | self._check_pos(batch_us, candidates)

                negs = np.empty((n, self.nnegs), dtype=np.int64)

                order = np.argsort(invalid, axis=1, kind='stable')
                sorted_candidates = candidates[np.arange(n)[:, None], order]
                negs = sorted_candidates[:, :self.nnegs]
                n_valid = (~invalid).sum(axis=1)  # (n,)
                short = np.where(n_valid < self.nnegs)[0]
                for i in short:
                    n_missing = self.nnegs - n_valid[i]
                    negs[i, n_valid[i]:] = self.rng.integers(0, self.N, size=n_missing)
                ix[mask, 2:] = negs
            yield torch.from_numpy(ix), pos_lists

    def __len__(self):
        return int(np.ceil(len(self.edges) / self.batch_size))
