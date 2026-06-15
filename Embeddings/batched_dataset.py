import numpy as np
import random
import torch
import scipy
import networkx as nx
from collections import deque, defaultdict
from numpy.random import default_rng
from tqdm import tqdm


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
        self, edges, objects, nnegs, batch_size, burnin=False, 
        depth_temperature=1.0, max_edges_per_epoch=None
        ):
        self.edges = edges
        self.edges_arr = np.array(edges, dtype=np.int64)
        self.max_edges_per_epoch = max_edges_per_epoch
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
        self.pos_neighbors_arr = [np.array(list(s), dtype=np.int64) for s in self.pos_neighbors]
        rows = self.edges_arr[:, 0]
        cols = self.edges_arr[:, 1]
        self.pos_matrix = scipy.sparse.csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(self.N, self.N))
        
        # Profondeurs
        edge_index = torch.tensor(edges, dtype=torch.long).T
        self.depths = compute_depths(edge_index, self.N).numpy()
        self._build_alias_tables()
        self._build_reachable()

    def _build_reachable(self, max_depth=2):
        G = nx.DiGraph()
        G.add_edges_from(self.edges_arr)
        self.reachable = {}
        for u in range(self.N):
            reached = nx.single_source_shortest_path_length(G, u, cutoff=max_depth)
            self.reachable[u] = np.array([v for v, d in reached.items() if d > 0], dtype=np.int64)

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
            
            ix = np.empty((B, 2 + self.nnegs), dtype=np.int64)
            ix[:, 0] = us
            ix[:, 1] = vs

            target_depths = self.depths[vs]

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
            yield torch.from_numpy(ix)

    def __len__(self):
        return int(np.ceil(len(self.edges) / self.batch_size))


class BatchedDatasetNode2Vec:
    def __init__(self, nx_G, is_directed, p, q, batchsize, nnegs, window_size, refresh_every):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q
        self.batchsize = batchsize
        self.nnegs = nnegs
        self.window_size = window_size
        self.refresh = refresh_every

        self.nodes = sorted(nx_G.nodes())
        self.node2idx = {n: i for i, n in enumerate(self.nodes)}
        self.N = len(self.nodes)

        # Voisins positifs indexés (pour exclure du sampling négatif)
        self.pos_neighbors: list[set] = [set() for _ in range(self.N)]
        for u, v in nx_G.edges():
            ui, vi = self.node2idx[u], self.node2idx[v]
            self.pos_neighbors[ui].add(vi)
            if not is_directed:
                self.pos_neighbors[vi].add(ui)

        self.rng = np.random.default_rng()
        self._epoch = 0
        self._active_pairs: np.ndarray | None = None
    
    @staticmethod
    def alias_setup(probs):
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=int)
        smaller, larger = [], []
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            (smaller if q[kk] < 1.0 else larger).append(kk)
        while smaller and larger:
            small = smaller.pop()
            large = larger.pop()
            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            (smaller if q[large] < 1.0 else larger).append(large)
        return J, q

    @staticmethod
    def alias_draw(J, q):
        K = len(J)
        kk = int(np.floor(np.random.rand() * K))
        return kk if np.random.rand() < q[kk] else J[kk]

    def node2vec_walk(self, walk_length, start_node):
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if not cur_nbrs:
                break
            if len(walk) == 1:
                idx = self.alias_draw(*alias_nodes[cur])
            else:
                prev = walk[-2]
                idx = self.alias_draw(*alias_edges[(prev, cur)])
            walk.append(cur_nbrs[idx])
        return walk

    def simulate_walks(self, num_walks, walk_length, verbose=True):
        walks = []
        nodes = list(self.G.nodes())
        for i in range(num_walks):
            if verbose:
                print(f"Walk {i+1}/{num_walks}")
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length, node))
        return walks
    
    def get_alias_edge(self, src, dst):
        G, p, q = self.G, self.p, self.q
        probs = []
        for nbr in sorted(G.neighbors(dst)):
            w = G[dst][nbr]['weight']
            if nbr == src:
                probs.append(w / p)
            elif G.has_edge(nbr, src):
                probs.append(w)
            else:
                probs.append(w / q)
        s = sum(probs)
        return self.alias_setup([x / s for x in probs])
    
    def preprocess_transition_probs(self):
        G = self.G, self.is_directed
        alias_nodes = {}
        for node in tqdm(G.nodes(), total=len(G.nodes())):
            probs = [G[node][nbr].get('weight', 1.0) for nbr in sorted(G.neighbors(node))]
            s = sum(probs)
            alias_nodes[node] = self.alias_setup([x / s for x in probs])
        self.alias_nodes = alias_nodes
        self.alias_edges = PrecomputeAliasEdges(self)
            #alias_edges[edge] = self.get_alias_edge(*edge)
            #if not is_directed:
                #alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        # self.alias_edges = alias_edges

    def _walks_to_pairs(self, walks):
        """
        Extrait toutes les paires (centre, contexte) pour construire un ensemble
        de paires positives (parmi lesquelles ne pas tirer de négatif).
        """
        pairs = []
        W = self.window_size
        for walk in walks:
            walk_idx = [self.node2idx[n] for n in walk]
            L = len(walk_idx)
            for i, u in enumerate(walk_idx):
                lo = max(0, i - W)
                hi = min(L, i + W + 1)
                for j in range(lo, hi):
                    if j != i:
                        pairs.append((u, walk_idx[j]))
        return np.array(pairs, dtype=int)

    def _sample_negatives(self, u: int, positive_context):
        """
        Tire nnegs indices négatifs pour le nœud u.
        Exclut les voisins positifs et u lui-même.
        """
        excluded = self.pos_neighbors[u] | {u} | positive_context
        negs = []
        while len(negs) < self.nnegs:
            candidates = self.rng.integers(0, self.N, size=self.nnegs * 3)
            for c in candidates:
                if c not in excluded:
                    negs.append(c)
                if len(negs) == self.nnegs:
                    break
        return np.array(negs[:self.nnegs], dtype=int)

    def _build_batch(self, pairs: np.ndarray) -> torch.Tensor:
        idx = self.rng.choice(len(pairs), size=self.batchsize, replace=False)
        selected = pairs[idx]          # (B, 2)
        batch = np.zeros((self.batchsize, 1 + self.nnegs, 2), dtype=np.int32)
        batch[:, 0] = selected         # positifs
        for i, (u, _) in enumerate(selected):
            negs = self._sample_negatives(int(u))
            batch[i, 1:, 0] = u        # même source
            batch[i, 1:, 1] = negs
        return torch.tensor(batch, dtype=torch.long)

    def epoch_batches(self, num_walks: int, walk_length: int):
        """
        Yield des tenseurs de taille (B, 1+nnegs) pour une epoch contenant chaque noeud positif
        et ces noeuds négatifs associés.
        Recalcule les paires si nécessaire (premier appel ou refresh).
        """
        if self._active_pairs is None or self._epoch % self.refresh == 0:
            walks = self.simulate_walks(num_walks, walk_length, verbose=False)
            self._active_pairs = self._walks_to_pairs(walks)
        
        pairs = self._active_pairs
        n_batches = len(pairs) // self.batchsize
        perm = self.rng.permutation(len(pairs))
        pairs = pairs[perm]

        for i in range(n_batches):
            chunk = pairs[i * self.batchsize:(i + 1) * self.batchsize]
            batch = np.zeros((self.batchsize, 2 + self.nnegs), dtype=int)
            batch[:, 0] = chunk[:, 0]
            batch[:, 1] = chunk[:, 1]
            for j, (u, v_pos) in enumerate(chunk):
                pos_context = set(chunk[chunk[:, 0] == u][:, 1].tolist())  # Tous les noeuds qui ne sont pas dans la fenêtre
                negs = self._sample_negatives(int(u), pos_context)
                batch[j, 2:] = negs
            yield torch.tensor(batch, dtype=torch.long)

        self._epoch += 1

    def __len__(self):
        return int(np.ceil(len(self.edges) / self.batch_size))


class PrecomputeAliasEdges:
    def __init__(self, G):
        self.G = G
        self._cache = {}
    
    def __getitem__(self, edge):
        if edge not in self._cache:
            self._cache[edge] = self.G.get_alias_edge(*edge)
        return self._cache[edge]
    
    def __contains__(self, edge):
        return True
