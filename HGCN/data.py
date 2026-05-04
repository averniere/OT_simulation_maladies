import networkx as nx
import scipy.sparse
import numpy as np
import torch


def load_data(args, hpo_graph, omim_df):
    # adjacency matrix
    print("Load HPOs")
    nodes = list(hpo_graph.nodes())
    node2idx = {n: i for i, n in enumerate(nodes)}
    adj = nx.to_scipy_sparse_array(hpo_graph, nodelist=nodes, format='csr')
    
    # Features
    print("Load features")
    hpo_cols = [col for col in omim_df.columns if col in node2idx]
    features = np.zeros((len(nodes), len(omim_df)))
    for col in hpo_cols:
        idx = node2idx[col]
        features[idx, :] = omim_df[col].values

    #omim_reindexed = omim_df[sorted(hpo_cols, key=lambda x: node2idx[x])]
    
    features = scipy.sparse.csr_matrix(features)  # (termes x maladies)
    print(adj.shape)       # doit être (n_terms, n_terms)
    print(features.shape)  # doit être (n_terms, n_diseases)
    
    data = {'adj_train': adj,'features': features}
    print("Load train, test and validation datasets")
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                    adj, args.val_prop, args.test_prop, args.split_seed)
    data['adj_train'] = adj_train
    data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
    data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
    data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
    
    data['adj_train_norm'], data['features'] = process(
        data['adj_train'], data['features'], 
        args.normalize_adj, args.normalize_feats)
    return data



def mask_edges(adj, val_prop, test_prop, seed):
    # Positive edges (voisins)
    print("Load positive edges")
    np.random.seed(seed)
    x, y = scipy.sparse.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # Negative edges (non-voisins)
    print("Load negative edges")
    #x, y = scipy.sparse.triu(scipy.sparse.csr_matrix(1. - adj.toarray())).nonzero()

    def sample_neg_edges(adj, n_samples, seed):
        np.random.seed(seed)
        n = adj.shape[0]
        neg_edges = []
        existing = set(zip(*adj.nonzero()))
        while len(neg_edges) < n_samples:
            i, j = np.random.randint(0, n, 2)
            if i < j and (i, j) not in existing:
                neg_edges.append([i, j])
        return np.array(neg_edges)
    #neg_edges = np.array(list(zip(x, y)))
    neg_edges = sample_neg_edges(adj, len(pos_edges), seed)
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    #train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    train_nodes = set(train_edges[:, 0].tolist()) | set(train_edges[:, 1].tolist())
    train_neg_mask = np.array([e[0] in train_nodes and e[1] in train_nodes for e in neg_edges])
    train_edges_false = np.concatenate([neg_edges[train_neg_mask], val_edges, test_edges], axis=0)

    adj_train = scipy.sparse.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false)  


def process(adj, features, normalize_adj, normalize_feats):
    if scipy.sparse.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    zero_rows = (features.sum(dim=1) == 0)
    print(f"Noeuds avec features nulles : {zero_rows.sum().item()}")
    features[zero_rows] = torch.randn(zero_rows.sum(), features.shape[1]) * 1e-5
    if normalize_adj:
        adj = normalize(adj + scipy.sparse.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1), dtype=np.float64)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = scipy.sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
