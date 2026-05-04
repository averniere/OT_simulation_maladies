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
import heapq


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
		isolated_nodes = set([431, 558, 1028, 1098, 1101, 1102])
		for d, i, j in edges:
			iterations += 1
			if uf.union(i, j):
				value = max(d, 1e-4)
				KNN[i, j] = value
				KNN[j, i] = value
				# Debug
				if i in isolated_nodes or j in isolated_nodes:
					node = i if i in isolated_nodes else j
					other = j if i in isolated_nodes else i
					print(f"Nœud isolé {node} fusionné avec {other}, d={d}, value={value}, KNN[{node},{other}]={KNN[node,other]}")
        
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


"""	
	cur_comp = 0
	iteration = 0
	max_iterations = n_components
	with tqdm(total=max_iterations, desc="Connexion des composantes", unit="itération") as pbar:
		while n_components > 1:
			iteration += 1
			idx_cur = np.where(labels == cur_comp)[0]
			idx_rest = np.where(labels != cur_comp)[0]
			d = distances[idx_cur][:, idx_rest]
			ia, ja = np.where(d == np.min(d))
			i = ia
			j = ja

			KNN[idx_cur[i], idx_rest[j]] = distances[idx_cur[i], idx_rest[j]]
			KNN[idx_rest[j], idx_cur[i]] = distances[idx_rest[j], idx_cur[i]]

			nearest_comp = labels[idx_rest[j]][0]
			labels[labels == nearest_comp] = cur_comp
			c = [list(labels).count(x) for x in np.unique(labels)]
			# if n_components % 100 == 0 :
				# print("Nombre de composantes :", len(c))
			n_components = len(c)
			pbar.update(1)
			if iteration > max_iterations:
				print("Attention : Nombre maximum d'itérations atteint.")
				print(len(c))
	n_comp_final, _ = connected_components(KNN, directed=False)
	print("Nombre de composantes après la boucle while :", n_comp_final)
	return KNN
"""

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
	if mode == 'features':
		KNN_sparse = kneighbors_graph(features, k_neighbours, mode='distance', metric=distlocal, include_self=False, n_jobs=-1)
		KNN_sparse.data[KNN_sparse.data == 0.0] = 1e-4
		row = KNN_sparse.getrow(429)
		print(f"Voisins de 429 : {row.indices}")
		print(f"Distances : {row.data}")

		KNN = KNN_sparse.toarray()

		if sym:
			KNN = np.maximum(KNN, KNN.T)  # Dès qu'un noeud est voisin d'un autre on l'inclue dans le graphe
		else:
			KNN = np.minimum(KNN, KNN.T)  # On ne garde que les noeuds qui sont voisins l'un de l'autre

		n_components, labels = csgraph.connected_components(KNN, directed=False)
		print("n_components KNN", n_components)
		
		old_duplicates = [431, 558, 1028, 1098, 1101, 1102]  # etc.
		for node in old_duplicates:
			print(f"Nœud {node} → composante {labels[node]}, même que 429 ({labels[429]}) : {labels[node] == labels[429]}")

		if connected and (n_components > 1):
			distances = pairwise_distances(features, metric=distlocal, n_jobs=-1)
			print(f"NaN dans distances : {np.isnan(distances).sum()}")
			print(f"Inf dans distances : {np.isinf(distances).sum()}")

			isolated = np.where(KNN.sum(axis=1) == 0)[0]
			print(f"Nœuds sans aucun voisin dans KNN : {len(isolated)}")
			print(f"Distances min/max de ces nœuds : {distances[isolated].min()}, {distances[isolated].max()}")
			
			KNN, labels_new, n_components_new = connect_knn(KNN, distances, n_components, labels)
			print(f"Non nuls juste après connect_knn : {np.count_nonzero(KNN)}")
			print(f"Isolés juste après connect_knn : {len(np.where(KNN.sum(axis=1) == 0)[0])}")
	else:
		KNN = features

	# Diagnostic I
	n_comp_final, comp_labels = connected_components(KNN, directed=True)

	print(f"Composantes connexes avant shortest_path : {n_comp_final}")
	print(f"Valeurs non nulles dans KNN : {np.count_nonzero(KNN)}")

	# Diagnostic II
	comp_sizes = np.bincount(comp_labels)
	sorted_comp = np.argsort(comp_sizes)  # du plus petit au plus grand
	print(f"{n_comp_final} composantes :")
	for rank, comp_id in enumerate(sorted_comp):
		nodes = np.where(comp_labels == comp_id)[0]
		print(f"  Composante {rank+1} (id={comp_id}) : {len(nodes)} nœuds → {nodes[:10]}{'...' if len(nodes) > 10 else ''}")

	D_high = shortest_path(KNN, method="D", directed=False)
	print("Nb inf dans D_high :", np.isinf(D_high).sum())

	if distlocal == 'minkowski':
		S = np.exp(-KNN / (sigma*features.size(1)))
	else:
		S = np.exp(-KNN / sigma)

	S[KNN == 0] = 0
	L = csgraph.laplacian(S, normed=False)

	n = L.shape[0]
	RFA = solve(L + np.eye(n), np.eye(n), assume_a='pos')
	RFA[RFA == np.nan] = 0.0

	return KNN, D_high