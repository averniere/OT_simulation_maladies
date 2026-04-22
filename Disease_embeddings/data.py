from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from sklearn.utils.graph_shortest_path import graph_shortest_path
from scipy.sparse import csgraph

import numpy as np
import torch


def prepare_data(df, with_labels=True, normalize=False, n_pca=0):
	n = len(df.columns)

	if with_labels:
		x = np.double(df.values[:, 1:n])
		labels = df.values[:, 0]
		labels = labels.astype(str)
		colnames = df.columns[1:n]
	else:
		x = np.double(df.values)
		labels = ['unknown'] * np.size(x, 0)
		colnames = df.columns

	n = len(colnames)

	idx = np.where(np.std(x, axis=0) != 0)[0]
	x = x[:, idx]

	if normalize:
		s = np.std(x, axis=0)
		s[s == 0] = 1
		x = (x - np.mean(x, axis=0)) / s

	if n_pca:
		if n_pca == 1:
			n_pca = n

		nc = min(n_pca, n)
		pca = PCA(n_components=nc)
		x = pca.fit_transform(x)
	labels = np.array([str(s) for s in labels])
	return torch.DoubleTensor(x), labels


def connect_knn(KNN, distances, n_components, labels):
	"""
	Given a KNN graph, connect nodes until we obtain a single connected
	component.
	"""
	c = [list(labels).count(x) for x in np.unique(labels)]

	cur_comp = 0
	while n_components > 1:
		idx_cur = np.where(labels == cur_comp)[0]
		idx_rest = np.where(labels != cur_comp)[0]
		d = distances[idx_cur][:, idx_rest]
		min_idx = np.argmin(d)
		i, j = np.unravel_index(min_idx, d.shape)

		KNN[idx_cur[i], idx_rest[j]] = distances[idx_cur[i], idx_rest[j]]
		KNN[idx_rest[j], idx_cur[i]] = distances[idx_rest[j], idx_cur[i]]

		nearest_comp = labels[idx_rest[j]]
		labels[labels == nearest_comp] = cur_comp
		n_components -= 1

	return KNN


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
		KNN = kneighbors_graph(features, k_neighbours, mode='distance', metric=distlocal, include_self=False).toarray()

		if sym:
			KNN = np.maximum(KNN, KNN.T)  # Dès qu'un noeud est voisin d'un autre on l'inclue dans le graphe
		else:
			KNN = np.minimum(KNN, KNN.T)  # On ne garde que les noeuds qui sont voisins l'un de l'autre

		n_components, labels = csgraph.connected_components(KNN)

		if connected and (n_components > 1):
			from sklearn.metrics import pairwise_distances
			distances = pairwise_distances(features, metric=distlocal)
			KNN = connect_knn(KNN, distances, n_components, labels)
	else:
		KNN = features

	D_high = graph_shortest_path(KNN)

	if distlocal == 'minkowski':
		S = np.exp(-KNN / (sigma*features.size(1)))
	else:
		S = np.exp(-KNN / sigma)

	S[KNN == 0] = 0
	L = csgraph.laplacian(S, normed=False)

	RFA = np.linalg.inv(L + np.eye(L.shape[0]))
	RFA[RFA == np.nan] = 0.0

	return torch.Tensor(RFA), D_high