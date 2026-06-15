import numpy as np
import networkx as nx
import random

from gensim.models import Word2Vec
from data import *
from data_utils import add_corresponding_terms, add_edges

class Graph():
	def __init__(self, nx_G, is_directed, p, q,): # batchsize, nnegs, window_size, refresh_every):
		self.G = nx_G
		self.is_directed = is_directed
		self.p = p
		self.q = q
		#self.batchsize = batchsize
		#self.nnegs = nnegs
		#self.window_size = window_size
		#self.refresh = refresh_every
		self.nodes = sorted(nx_G.nodes())
		self.node2idx = {n: i for i, n in enumerate(self.nodes)}
		self.N = len(self.nodes)
		self.pos_neighbors: list[set] = [set() for _ in range(self.N)]
		for u, v in nx_G.edges():
			ui, vi = self.node2idx[u], self.node2idx[v]
			self.pos_neighbors[ui].add(vi)
			if not is_directed:
				self.pos_neighbors[vi].add(ui)
		#self.rng = np.random.default_rng()
		self._epoch = 0
		#self._active_pairs: np.ndarray | None = None
		
	@staticmethod
	def alias_setup(probs):
		K = len(probs)
		q = np.zeros(K)
		J = np.zeros(K, dtype=np.int)
		smaller = []
		larger = []
		for kk, prob in enumerate(probs):
			q[kk] = K*prob
			if q[kk] < 1.0:
				smaller.append(kk)
			else:
				larger.append(kk)
		while len(smaller) > 0 and len(larger) > 0:
			small = smaller.pop()
			large = larger.pop()
			J[small] = large
			q[large] = q[large] + q[small] - 1.0
			if q[large] < 1.0:
				smaller.append(large)
			else:
				larger.append(large)
		return J, q
	
	@staticmethod
	def alias_draw(J, q):
		'''
		Draw sample from a non-uniform discrete distribution using alias sampling.
		'''
		K = len(J)
		kk = int(np.floor(np.random.rand()*K))
		if np.random.rand() < q[kk]:
			return kk
		else:
			return J[kk]

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next = cur_nbrs[self.alias_draw(alias_edges[(prev, cur)][0], 
						alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break

		return walk

	def simulate_walks(self, num_walks, walk_length, verbose=True):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())
		if verbose:
			print('Walk iteration:')
		for walk_iter in range(num_walks):
			if verbose:
				print(str(walk_iter+1), '/', str(num_walks))
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

		return walks

	def get_alias_edge(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self.G
		p = self.p
		q = self.q

		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
		norm_const = sum(unnormalized_probs)
		normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return self.alias_setup(normalized_probs)

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G
		is_directed = self.is_directed

		alias_nodes = {}
		for node in G.nodes():
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			alias_nodes[node] = self.alias_setup(normalized_probs)

		alias_edges = {}
		triads = {}

		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges
		return 

print('Union diseases')
union_diseases = add_corresponding_terms(work_omim, work_orpha, df_orpha_omim)
print('Add edges')
G_hpo_omim = add_edges(union_diseases, G_hpo_work, depths)
print("End of preprocessing")

np.random.seed(0)
# adj_sparse = nx.to_scipy_sparse_matrix(G_hpo_omim)

P = 1 # Return hyperparameter
Q = 0.05 # In-out hyperparameter
WINDOW_SIZE = 10 # Context size for optimization
NUM_WALKS = 10 # Number of walks per source
WALK_LENGTH = 5 # Length of walk per source
DIMENSIONS = 128 # Embedding dimension
DIRECTED = False # Graph directed/undirected
WORKERS = 8 # Num. parallel workers
ITER = 1 # SGD epochs

# Preprocessing, generate walks
g_n2v = Graph(G_hpo_omim, DIRECTED, P, Q)
g_n2v.preprocess_transition_probs()
walks = g_n2v.simulate_walks(NUM_WALKS, WALK_LENGTH)
walks = [list(map(str, walk)) for walk in walks]

print(walks)

# Train skip-gram model
model = Word2Vec(walks, size=DIMENSIONS, window=WINDOW_SIZE, min_count=0, sg=1, workers=WORKERS, iter=ITER)
emb_mappings = model.wv
model.wv.save_word2vec_format('HPOEmb-Orphanet.emd')
