import networkx as nx
import numpy as np
import numba
from collections import deque
from scipy.spatial.distance import cdist
from tqdm import tqdm
from data import *
from information_content import get_ancestors0, compute_information_content


class tsw:
    def __init__(self, G, weights, root='HP:0000001'):
        '''
        G : G_hpo_work (orienté dans le sens enfant --> parent)
        weights : pondérations des arêtes, soit uniforme, soit avec l'ic, soit avec les profondeurs
        sous la forme d'un dictionnaire {'HP:..':...}.
        '''
        self.G = G
        self.root = root
        self.weights = weights
        self.nodes = G.nodes()
        self.n = len(self.nodes)
        self.node2id = {n: i for i, n in enumerate(self.nodes)}
        self.parents = [list(G.successors(node)) for node in self.nodes]
        self.children = [list(G.predecessors(node)) for node in self.nodes]
        self.order = self._sort(root)
        self.order_indices = np.array([self.node2id[node] for node in self.order])
        edge_weight = []
        for node in self.order:
            edge_weight.append(weights[node])
        self.edge_weight = edge_weight
    
    def _build_numba_structures(self):
        """Convertit parents et order en arrays numpy continus."""
        n = self.n
        
        self.order_idx = self.order_indices.astype(np.int32)
        parents_ptr = np.zeros(n + 1, dtype=np.int32)
        for k in range(n):
            parents_ptr[k + 1] = parents_ptr[k] + len(self.parents[k])
        parents_flat = np.empty(parents_ptr[n], dtype=np.int32)
        for k in range(n):
            for j, p in enumerate(self.parents[k]):
                parents_flat[parents_ptr[k] + j] = self.node2id[p]
        
        self.parents_ptr = parents_ptr
        self.parents_flat = parents_flat
        
        self.root_idx = np.int32(self.node2id[self.root])
        
        self.edge_weight_arr = np.array(self.edge_weight, dtype=np.float32)

    def _sort(self, root):
        '''
        Retourne les noeuds dans l'ordre de remontée dans l'arbre.
        '''
        visited = set()
        q = deque()
        q.append(root)
        seen = set()
        visited.add(root)
        bfs = []
        while q:
            node = q.popleft()
            node_idx = self.node2id[node]
            bfs.append(node)
            for child in self.children[node_idx]:
                if child not in seen:
                    seen.add(child)
                    q.append(child)
        return list(reversed(bfs))

    def prepare_disease(self, diseases):
        '''
        Input:
            - diseases : dataframe avec les maladies de taille (n, m').
        Output:
            - new_M : ndarray de taille (n, m) encodé sur les d noeuds du graphe, de valeurs 0 ou 1/n_actif.
        '''
        hpo_cols = [c for c in diseases.columns if c.startswith('HP')]
        hpo_ids = [self.node2id[hpo] for hpo in hpo_cols]
        M = diseases.values
        d = M.shape[0]  # Nombre de maladies
        new_M = np.zeros((d, self.n), dtype=np.int32)
        for i in range(d):
            for j in range(len(hpo_cols)):
                if M[i, j]==1:
                    idx = hpo_ids[j]
                    new_M[i, idx] = 1
        new_M = new_M/np.sum(new_M, axis=1, keepdims=True)
        return new_M

    def f_tsw(self, d1, d2):
        '''
        Input:
            - d1, d2 : deux maladies sous la forme d'arrays numpy avec pour valeurs soit 1, soit 1/n_actif
        Output:
            - distance de Wasserstein dans l'arbre, selon la propriété 1 de l'article Tree-Sliced Variants of Wasserstein Distances.
        '''
        print("Subtree mass")
        mu1, mu2 = self.subtree_mass(d1), self.subtree_mass(d2)  # mu(Gamma(ve)), nu(Gamma(ve)) pour tout e
        #diff = np.abs(mu1-mu2)
        diff = np.abs(mu1[self.order_indices] - mu2[self.order_indices])
        print("Dot product")
        return np.float32(np.dot(self.edge_weight, diff))

    def tsw_matrix(self, diseases1, diseases2, batchsize=256):
        '''
        Input : 
            - diseases : dataframe (n, K) de maladies.
        Retourne une matrice (n, n) de distances entre maladies.
        '''
        M1 = self.prepare_disease(diseases1).astype(np.float32)
        M2 = self.prepare_disease(diseases2).astype(np.float32)
        N1, N2 = M1.shape[0], M2.shape[0]
        C = np.zeros((N1, N2), dtype=np.float32)
        w = self.edge_weight_arr

        SM2 = subtree_mass(
            M2.copy(), self.order_idx,
            self.parents_ptr, self.parents_flat, self.root_idx
        )[:, self.order_idx]

        for i in tqdm(range(0, N1, batchsize)):
            i_end = min(N1, i + batchsize)
            SM1_batch = subtree_mass(
                M1[i:i_end].copy(), self.order_idx,
                self.parents_ptr, self.parents_flat, self.root_idx
            )[:, self.order_idx]  # (batch, E)
            
            C[i:i_end] = precompute_tsw_matrix(SM1_batch, SM2, w)
                #C[i, j] = self.f_tsw(M1[i], M2[j])
        return C


@numba.njit(parallel=False, cache=True)
def subtree_mass(mu, order_idx, parents_ptr, parents_flat, root_idx):
        '''
        Input: 
            - disease : batch de maladies sous la forme d'un vecteur binaire de taille N,d.
        Output:
            - mu : batch de vecteurs de taille d, tel que mu[i] = masse totale sous le noeud i.
        '''
        N = mu.shape[0]

        #for node in self.order:
        for ind in range(len(order_idx)):
            k = order_idx[ind]
            start = parents_ptr[k]
            end = parents_ptr[k+1]
            for pi in range(start, end):
                p = parents_flat[pi]
                if p != root_idx:
                    for i in range(N):
                        mu[i, p] += mu[i, k]
            #k = self.node2id[node]
            #p_list = self.parents[k]
            #if len(p_list) > 0:
                #for p in p_list:
                    #if p != self.root:
                        #i = self.node2id[p]
                        #mu[:, i] += mu[:, k]
        return mu


@numba.njit(parallel=True, cache=True)
def precompute_tsw_matrix(SM1, SM2, w):
    N1 = SM1.shape[0]
    N2 = SM2.shape[0]
    E  = SM1.shape[1]
    C  = np.zeros((N1, N2), dtype=np.float32)
    for i in numba.prange(N1):          # parallèle sur N1
        for j in range(N2):
            s = np.float32(0.0)
            for e in range(E):
                s += w[e] * abs(SM1[i, e] - SM2[j, e])
            C[i, j] = s
    return C

uniform = {node: 1.0 for node in G_hpo_work.nodes()}

tree = tsw(G_hpo_work, uniform)
tree._build_numba_structures()
C = tree.tsw_matrix(work_omim, work_orpha)
print(C)

