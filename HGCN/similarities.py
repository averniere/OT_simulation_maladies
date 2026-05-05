from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csgraph
from scipy.linalg import solve
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from tqdm import tqdm

import numpy as np
import torch
import hashlib
import pickle
import os


def prepare_data(df, with_labels=True, normalize=False, n_pca=0):

	n = len(df.columns)
	if with_labels:
		x = np.double(df.values[:, 1:])
		labels = df.values[:, 0]
		labels = labels.astype(str)
		colnames = df.columns[1:]
	else:
		x = np.double(df.values)
		labels = ['unknown'] * np.size(x, 0)
		colnames = df.columns

	idx = np.where(np.std(x, axis=0) != 0)[0]
	x = x[:, idx]

	if normalize:
		s = np.std(x, axis=0)
		s[s == 0] = 1
		x = (x - np.mean(x, axis=0)) / s

	if n_pca:
		if n_pca == 1:
			n_pca = n
		nc = min(n_pca, x.shape[0], x.shape[1])
		pca = PCA(n_components=nc)
		x = pca.fit_transform(x)
	labels = np.array([str(s) for s in labels])
	return x, torch.DoubleTensor(x), labels


class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)
        self.rank = np.zeros(n, dtype=int)
        self.n_components = n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Déjà dans la même composante
        
        # Union par rang pour garder l'arbre plat
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
            
        self.n_components -= 1
        return True


def connect_knn(KNN, distances, n_components, labels):
	"""
	Given a KNN graph, connect nodes until we obtain a single connected
	component.
	"""	
	n_nodes = KNN.shape[0]
	uf = UnionFind(n_nodes)
	first_node_per_label = {}
	for i in range(n_nodes):
		lbl = labels[i]
		if lbl not in first_node_per_label:
			first_node_per_label[lbl] = i
		else:
			uf.union(i, first_node_per_label[lbl])
	print(f"Initialisation Union-Find. Composantes initiales : {uf.n_components}")
	edges = []
	for i in range(n_nodes):
		for j in range(i + 1, n_nodes):
			d = distances[i, j]
			edges.append((d, i, j))
	edges.sort(key=lambda x: x[0])
	iterations = 0
	total_edges_checked = len(edges)
	with tqdm(total=total_edges_checked, desc="Kruskal (Tri & Fusion)", unit="arête") as pbar:
		for d, i, j in edges:
			iterations += 1
			if uf.union(i, j):
				value = max(d, 1e-4)
				KNN[i, j] = value
				KNN[j, i] = value
				
				if uf.n_components == 1:
					print("Toutes les composantes sont connectées.")
					pbar.update(1)
					break
			pbar.update(1)
	print(f"Terminé. Arêtes traitées : {iterations}, Composantes finales : {uf.n_components}")
	n_comp_check, _ = csgraph.connected_components(KNN, directed=True)
	print(f"Vérification connectivité réelle de KNN : {n_comp_check} composantes")
	final_labels = np.array([uf.find(i) for i in range(n_nodes)])
	return KNN, final_labels, uf.n_components


def compute_rfa(features, mode='features', k_neighbours=15, sym=False, connected=False, sigma=1.0, distlocal='minkowski'):
	"""
	Computes the target RFA similarity matrix. The RFA matrix of
	similarities relates to the commute time between pairs of nodes, and it is
	built on top of the Laplacian of a single connected component k-nearest
	neighbour graph of the data.
	Inputs : 
		- k_neighbours : nombre d'éléments par cluster.
		- sym : False si l'on souhaite conserver un graphe sparse avec seulement les voisins qui se désignent
		les uns les autres ou si l'on conserve le lien dès qu'un noeud est voisin d'un autre.
		- connected : True si l'on connecte les composantes connexes isolées à l'issu du clustering.
		- sigma : hyperparamètre qui contrôle l'écart type de la transformation gaussienne des poids.
		- distlocal : distance utilisée pour calculer les plus proches voisin.
	"""
	def get_cache_path(k_neighbours, sigma, distlocal, sym, connected):
		"""Génère un nom de fichier unique basé sur les paramètres."""
		params = f"{k_neighbours}_{sigma}_{distlocal}_{sym}_{connected}"
		hash_str = hashlib.md5(params.encode()).hexdigest()[:8]
		return f"../cache/knn_{hash_str}.pkl"

	cache_path = get_cache_path(k_neighbours, sigma, distlocal, sym, connected)

	if os.path.exists(cache_path):
		print("Chargement RFA depuis le cache...")
		with open(cache_path, 'rb') as f:
			KNN, D_high = pickle.load(f)
		return KNN, D_high
	else:
		if mode == 'features':
			KNN_sparse = kneighbors_graph(features, k_neighbours, mode='distance', metric=distlocal, include_self=False, n_jobs=-1)
			KNN_sparse.data[KNN_sparse.data == 0.0] = 1e-4
			KNN = KNN_sparse.toarray()

			if sym:
				KNN = np.maximum(KNN, KNN.T)  # Dès qu'un noeud est voisin d'un autre on l'inclue dans le graphe
			else:
				KNN = np.minimum(KNN, KNN.T)  # On ne garde que les noeuds qui sont voisins l'un de l'autre

			n_components, labels = csgraph.connected_components(KNN, directed=False)
			print("n_components KNN", n_components)
		
			if connected and (n_components > 1):
				distances = pairwise_distances(features, metric=distlocal, n_jobs=-1)
				print(f"NaN dans distances : {np.isnan(distances).sum()}")
				print(f"Inf dans distances : {np.isinf(distances).sum()}")

				isolated = np.where(KNN.sum(axis=1) == 0)[0]
				print(f"Nœuds sans aucun voisin dans KNN : {len(isolated)}")
			
				KNN, labels_new, n_components_new = connect_knn(KNN, distances, n_components, labels)
				print(f"Non nuls juste après connect_knn : {np.count_nonzero(KNN)}")
				print(f"Isolés juste après connect_knn : {len(np.where(KNN.sum(axis=1) == 0)[0])}")
		else:
			KNN = features

		# Diagnostic I
		n_comp_final, comp_labels = connected_components(KNN, directed=True)

		print(f"Composantes connexes avant shortest_path : {n_comp_final}")
		print(f"Valeurs non nulles dans KNN : {np.count_nonzero(KNN)}")

	
		D_high = shortest_path(KNN, method="D", directed=False)
		print("Nb inf dans D_high :", np.isinf(D_high).sum())
		os.makedirs("../cache", exist_ok=True)
		with open(cache_path, 'wb') as f:
			pickle.dump((KNN, D_high), f)
		print(f"sauvegardé dans {cache_path}")

		return KNN, D_high


def connect_knn_fast(KNN_sparse, features, labels, metric='cosine'):
    """
    Version optimisée : évite pairwise_distances dense O(n²).
    On connecte les composantes isolées en cherchant uniquement
    les voisins inter-composantes via un KNN élargi.
    """
    n_nodes = features.shape[0]
    uf = UnionFind(n_nodes)

    # Initialisation depuis le KNN existant
    cx = KNN_sparse.tocoo()
    for i, j in zip(cx.row, cx.col):
        uf.union(i, j)

    # Pré-fusion par label (comme avant)
    first_node_per_label = {}
    for i in range(n_nodes):
        lbl = labels[i]
        if lbl not in first_node_per_label:
            first_node_per_label[lbl] = i
        else:
            uf.union(i, first_node_per_label[lbl])

    if uf.n_components == 1:
        return KNN_sparse

    print(f"Composantes à connecter : {uf.n_components}")

    # Stratégie : augmenter k progressivement pour trouver des ponts
    # sans jamais calculer la matrice de distances complète
    KNN_lil = KNN_sparse.tolil()

    k_bridge = min(50, n_nodes - 1)
    print(f"Recherche de ponts avec k={k_bridge}...")

    bridge_graph = kneighbors_graph(
        features, k_bridge,
        mode='distance', metric=metric,
        include_self=False, n_jobs=-1
    )
    bridge_graph.data[bridge_graph.data == 0.0] = 1e-4

    # Trier les arêtes candidates par distance (sparse → pas de O(n²))
    cx2 = bridge_graph.tocoo()
    # Construire seulement les arêtes i < j
    mask = cx2.row < cx2.col
    rows = cx2.row[mask]
    cols = cx2.col[mask]
    data = cx2.data[mask]

    # Tri par distance croissante
    order = np.argsort(data)
    rows, cols, data = rows[order], cols[order], data[order]

    print(f"Arêtes candidates à tester : {len(rows)}")

    added = 0
    for idx in range(len(rows)):
        i, j, d = int(rows[idx]), int(cols[idx]), float(data[idx])
        if uf.union(i, j):
            value = max(d, 1e-4)
            KNN_lil[i, j] = value
            KNN_lil[j, i] = value
            added += 1
            if uf.n_components == 1:
                print(f"Graphe connecté après {added} arêtes ajoutées.")
                break

    if uf.n_components > 1:
        print(f"⚠️  Toujours {uf.n_components} composantes après k={k_bridge}.")
        # Fallback : augmenter k encore
        # (rare, seulement si les composantes sont vraiment isolées)

    return KNN_lil.tocsr()


def compute_rfa_fast(features, mode='features', k_neighbours=15, sym=False, connected=False, sigma=1.0, distlocal='cosine'):
	
	def get_cache_path(k_neighbours, sigma, distlocal, sym, connected):
		"""Génère un nom de fichier unique basé sur les paramètres."""
		if n_pca > 0:
			params = f"{k_neighbours}_{sigma}_{distlocal}_{sym}_{connected}"
		else:
			params = f"{k_neighbours}_{sigma}_{distlocal}_{sym}_{connected}"
		hash_str = hashlib.md5(params.encode()).hexdigest()[:8]
		return f"../cache/rfa_{hash_str}.pkl"
	
	cache_path = get_cache_path(k_neighbours, sigma, distlocal, sym, connected)
	if os.path.exists(cache_path):
		print("Chargement RFA depuis le cache...")
		with open(cache_path, 'rb') as f:
			KNN, D_high = pickle.load(f)
	else:
		
		print("Calcul RFA...")
		if mode == 'features':
			print("Calcul KNN sparse...")
			KNN_sparse = kneighbors_graph(features, k_neighbours, mode='distance', metric=distlocal, include_self=False, n_jobs=-1)
			KNN_sparse.data[KNN_sparse.data == 0.0] = 1e-4
			
			if sym:
				KNN_sparse = KNN_sparse.maximum(KNN_sparse.T)
			else:
				KNN_sparse = KNN_sparse.minimum(KNN_sparse.T)
				KNN_sparse.eliminate_zeros()

			n_components, labels = connected_components(KNN_sparse, directed=False)
			print(f"Composantes connexes KNN : {n_components}")
			
			if connected and n_components > 1:
				node_labels = np.array([str(i) for i in range(KNN_sparse.shape[0])])
				KNN_sparse = connect_knn_fast(KNN_sparse, features, node_labels, metric=distlocal)
				n_comp_check, _ = connected_components(KNN_sparse, directed=False)
				print(f"Composantes après connexion : {n_comp_check}")
		else:
			KNN_sparse = sp.csr_matrix(features)
			
		print("Calcul shortest_path (sparse Dijkstra)...")
		D_high = shortest_path(KNN_sparse, method="D", directed=False)
		print(f"Nb inf dans D_high : {np.isinf(D_high).sum()}")
		
		os.makedirs("../cache", exist_ok=True)
		with open(cache_path, 'wb') as f:
			pickle.dump((RFA, D_high), f)
		print(f"RFA sauvegardé dans {cache_path}")
		
		return KNN_sparse, D_high
