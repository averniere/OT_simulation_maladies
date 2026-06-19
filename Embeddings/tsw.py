import networkx as nx
import numpy as np
import pandas as pd
from collections import deque
from data import *
from information_content import get_ancestors0, compute_information_content
ancestors = {hp : get_ancestors0(G_hpo_work, hp) for hp in hpo_cols0}

class tsw:
    def __init__(self, G, weights, root='HP:0000001'):
        '''
        G : G_hpo_work (orienté dans le sens enfant --> parent)
        weights : pondérations des arêtes, soit uniforme, soit avec l'ic, soit avec les profondeurs.
        '''
        self.G = G
        self.root = root
        self.weights = weights
        self.nodes = G.nodes()
        self.n = len(self.nodes)
        self.node2id = {n: i for i, n in enumerate(self.nodes)}
        self.depths = nx.single_source_shortest_path_length(G.reverse(), source=root)
        self.parents = [list(G.successors(node)) for node in self.nodes]
        self.children = [list(G.predecessors(node)) for node in self.nodes]
        self.order = self._sort(root)
        edge_weight = []
        for node in self.order:
            edge_weight.append(weights[node])
        self.edge_weight = edge_weight


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
            bfs.append(node)
            for child in self.children[node]:
                if child not in seen:
                    seen.add(child)
                    q.append(child)
        return list(reversed(bfs))

    def subtree_mass(self, disease):
        '''
        Input: 
            - disease : maladie sous la forme d'un vecteur binaire de taille d.
        Output:
            - mu : vecteur de taille d, tel que mu[i] = masse totale sous le noeud i.
        '''
        mu = disease.copy()
        for node in self.order:
            k = self.node2id[node]
            p_list = self.parents[node]
            if len(p_list) > 0:
                for p in p_list:
                    if p != self.root:
                        i = self.node2id[p]
                        mu[i] += mu[k]
        return mu

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
        new_M = new_M/np.sum(new_M, axis=1)
        return new_M

    def f_tsw(self, d1, d2):
        '''
        Input:
            - d1, d2 : deux maladies sous la forme d'arrays numpy avec pour valeurs soit 1, soit 1/n_actif
        Output:
            - distance de Wasserstein dans l'arbre, selon la propriété 1 de l'article Tree-Sliced Variants of Wasserstein Distances.
        '''
        mu1 , mu2 = self.subtree_mass(d1), self.subtree_mass(d2) # mu(Gamma(ve)), nu(Gamma(ve)) pour tout e
        diff = np.abs(mu1-mu2)
        return np.float32(np.dot(self.edge_weight, diff))

    def tsw_matrix(self, diseases, batchsize=256):
        '''
        Input : 
            - diseases : dataframe (n, K) de maladies.
        Retourne une matrice (n, n) de distances entre maladies.
        '''
        M = self.prepare_disease(diseases)
        N = M.shape[0]
        C = np.zeros((N, N), dtype=np.float32)

        subtrees = np.stack([self.subtree_mass(M[i, :]) for i in range(N)])

        for i in range(0, N, batchsize):
            i_end = min(i+batchsize, N)
            diff = np.abs(subtrees[i:i_end, np.newaxis, :]-subtrees[np.newaxis, :, :])
            C[i:i_end, :] = diff @ self.edge_weight

        return C






        










