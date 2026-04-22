from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from scipy.sparse import csgraph

import pandas as pd
import numpy as np
import torch
import os


def prepare_data(df, with_labels=True, normalize=False, n_pca=0):
	
	n = len(df.columns)

	if with_labels:
		x = np.double(df.values[:, 0:n - 1])
		labels = df.values[:, (n - 1)]
		labels = labels.astype(str)
		colnames = df.columns[0:n - 1]
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
		ia, ja = np.where(d == np.min(d))
		i = ia
		j = ja

		KNN[idx_cur[i], idx_rest[j]] = distances[idx_cur[i], idx_rest[j]]
		KNN[idx_rest[j], idx_cur[i]] = distances[idx_rest[j], idx_cur[i]]

		nearest_comp = labels[idx_rest[j]]
		labels[labels == nearest_comp] = cur_comp
		n_components -= 1

	return KNN


def compute_rfa(features, mode='features', k_neighbours=15, distfn='sym', 
    connected=False, sigma=1.0, distlocal='minkowski'):
	"""
	Computes the target RFA similarity matrix. The RFA matrix of
	similarities relates to the commute time between pairs of nodes, and it is
	built on top of the Laplacian of a single connected component k-nearest
	neighbour graph of the data.
	"""
	start = timeit.default_timer()
	if mode == 'features':
		KNN = kneighbors_graph(features,
							   k_neighbours,
							   mode='distance',
							   metric=distlocal,
							   include_self=False).toarray()

		if 'sym' in distfn.lower():
			KNN = np.maximum(KNN, KNN.T)
		else:
			KNN = np.minimum(KNN, KNN.T)    

		n_components, labels = csgraph.connected_components(KNN)

		if connected and (n_components > 1):
			from sklearn.metrics import pairwise_distances
			distances = pairwise_distances(features, metric=distlocal)
			KNN = connect_knn(KNN, distances, n_components, labels)
	else:
		KNN = features    

	if distlocal == 'minkowski':
		S = np.exp(-KNN / (sigma*features.size(1)))
	else:
		S = np.exp(-KNN / sigma)

	S[KNN == 0] = 0
	L = csgraph.laplacian(S, normed=False)

	RFA = np.linalg.inv(L + np.eye(L.shape[0]))
	RFA[RFA==np.nan] = 0.0

	return torch.Tensor(RFA)