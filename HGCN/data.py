import networkx as nx
import scipy.sparse
import numpy as np
import pandas as pd
import torch
import hashlib
import pickle
import os

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

from similarities import compute_rfa, compute_rfa_fast


def get_cache_key(hpo_graph, omim_df):
    """Génère une clé unique basée sur les données d'entrée."""
    # Hash du graphe (edges + nodes)
    graph_hash = hashlib.md5(
        str(sorted(hpo_graph.edges())).encode() +
        str(sorted(hpo_graph.nodes())).encode()
    ).hexdigest()[:8]
    
    # Hash du dataframe (shape + colonnes + valeurs)
    df_hash = hashlib.md5(pd.util.hash_pandas_object(omim_df).values.tobytes()).hexdigest()[:8]
    
    return f"{graph_hash}_{df_hash}"


def load_data(args, hpo_graph, omim_df, cache_dir=".cache"):

    os.makedirs(cache_dir, exist_ok=True)
    cache_key = get_cache_key(hpo_graph, omim_df)
    cache_path = os.path.join(cache_dir, f"load_data_{cache_key}.pkl")

    if os.path.exists(cache_path):
        print(f"[Cache] Chargement depuis {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # adjacency matrix
    print("Load HPOs")
    nodes = list(hpo_graph.nodes())
    node2idx = {n: i for i, n in enumerate(nodes)}
    adj = nx.to_scipy_sparse_array(hpo_graph, nodelist=nodes, format='csr')
    
    # Features
    print("Propagate HPO annotations")
    omim_df = propagate_annotations(omim_df, hpo_graph)
    all_hpo_in_graph = [n for n in nodes if n in omim_df.columns]
    missing_hpo = [n for n in nodes if n not in omim_df.columns]

    print(f"Termes HPO avec colonne dans omim_df : {len(all_hpo_in_graph)}")
    print(f"Termes HPO sans colonne (nœuds internes purs) : {len(missing_hpo)}")

    hpo_cols = [col for col in omim_df.columns if col in node2idx]
    transformer = TfidfTransformer()
    hpo_matrix = transformer.fit_transform(omim_df[hpo_cols].values)
    # omim_df[hpo_cols] = hpo_matrix.toarray()
    print("Load features")
    
    omim_features = np.zeros((len(nodes), len(omim_df)))
    # for i, col in enumerate(hpo_cols):
        # idx = node2idx[col]
        # omim_features[idx] = (hpo_matrix[:, i].toarray().ravel())
    for col in all_hpo_in_graph:
        idx = node2idx[col]
        omim_features[idx] = hpo_matrix[:, hpo_cols.index(col)].toarray().ravel()
    G_rev = hpo_graph.reverse()
    for node in nx.topological_sort(G_rev):  # du bas vers le haut
        idx = node2idx[node]
        if omim_features[idx].sum() == 0:
            children = list(hpo_graph.predecessors(node))
            if children:
                child_feats = np.array([omim_features[node2idx[c]] for c in children])
                if child_feats.sum() > 0:
                    omim_features[idx] = child_feats.mean(axis=0)

    print("Compute structural features")
    features = compute_structural_features(hpo_graph, nodes, node2idx, omim_features)
    features = scipy.sparse.csr_matrix(features)

    zero_nodes = (omim_features.sum(axis=1) == 0).sum()
    print(f"Nœuds avec features nulles : {zero_nodes} / {len(nodes)}")

    '''
    # TEST
    features_dense = features.copy()
    zero_rows = np.where(features_dense.sum(axis=1) == 0)[0]
    print(f"Noeuds sans features : {len(zero_rows)}")
    adj_csr = scipy.sparse.csr_matrix(adj)
    for node in zero_rows:
        neighbors = adj_csr.getrow(node).indices
        neighbor_feats = features_dense[neighbors]
        if neighbor_feats.sum() > 0:
            features_dense[node] = neighbor_feats.mean(axis=0)
        else:
            features_dense[node] = features_dense.mean(axis=0) * 1e-3
    '''

    #omim_reindexed = omim_df[sorted(hpo_cols, key=lambda x: node2idx[x])]
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
        adj, data['features'], 
        args.normalize_adj, args.normalize_feats)

    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[Cache] Sauvegardé dans {cache_path}")

    return data


def compute_structural_features(hpo_graph, nodes, node2idx, omim_features):
    n = len(nodes)
    
    omim_feat = omim_features  # (n x n_diseases)
    
    degree = np.array([hpo_graph.degree(n) for n in nodes]).reshape(-1, 1)
    
    root = [n for n, d in hpo_graph.in_degree() if d == 0]
    if root:
        depths = nx.single_source_shortest_path_length(hpo_graph, root[0])
        depth = np.array([depths.get(n, 0) for n in nodes]).reshape(-1, 1)
    else:
        depth = np.zeros((n, 1))
    
    n_descendants = np.array([len(nx.descendants(hpo_graph, node)) for node in nodes]).reshape(-1, 1)
    
    struct_features = np.hstack([degree, depth, n_descendants])
    struct_features = struct_features / (struct_features.max(axis=0) + 1e-8)

    if not isinstance(omim_features, np.ndarray):
        omim_features = np.array(omim_features)
    
    features = np.hstack([omim_feat, struct_features])
    return features


def mask_edges(adj, val_prop, test_prop, seed):
    # Positive edges (voisins)
    print("Load positive edges")
    np.random.seed(seed)
    x, y = scipy.sparse.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)

    # Negative edges (non-voisins)
    print("Load negative edges")

    def sample_neg_edges(adj, n_samples, seed_offset):
        np.random.seed(seed+seed_offset)
        n = adj.shape[0]
        neg_edges = []
        existing = set(zip(*adj.nonzero()))
        while len(neg_edges) < n_samples:
            i, j = np.random.randint(0, n, 2)
            if i < j and (i, j) not in existing:
                neg_edges.append([i, j])
        return np.array(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    n_train = m_pos-n_val-n_test

    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]

    train_edges_false = sample_neg_edges(adj, n_train, 1)
    test_edges_false = sample_neg_edges(adj, n_test, 2)
    val_edges_false = sample_neg_edges(adj, n_val, 3)
    
    adj_train = scipy.sparse.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train +adj_train.T
    return (adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(test_edges_false),)

    # train_nodes = set(train_edges[:, 0].tolist()) | set(train_edges[:, 1].tolist())
    # train_neg_mask = np.array([e[0] in train_nodes and e[1] in train_nodes for e in neg_edges])
    # train_edges_false = np.concatenate([neg_edges[train_neg_mask], val_edges, test_edges], axis=0)

    # adj_train = scipy.sparse.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    # adj_train = adj_train + adj_train.T
    # return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           # torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            # test_edges_false)  


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


def propagate_annotations(omim_df, hpo_graph):
    """
    Propage les annotations HPO vers tous les ancêtres.

    Parameters
    ----------
    omim_df : pd.DataFrame
        lignes = maladies
        colonnes = HPO
        valeurs = 0/1

    hpo_graph : networkx.DiGraph
        Graphe HPO orienté parent -> enfant

    Returns
    -------
    propagated_df : pd.DataFrame
    """
    hpo_cols = [c for c in omim_df.columns if c in hpo_graph.nodes]
    hpo2idx = {h:i for i,h in enumerate(hpo_cols)}

    # Ancestors indices
    ancestors_idx = {}

    print("Precomputing ancestor indices...")
    for h in hpo_cols:
        anc = nx.ancestors(hpo_graph, h)
        anc_idx = [hpo2idx[a] for a in anc if a in hpo2idx]
        ancestors_idx[hpo2idx[h]] = anc_idx

    X = omim_df[hpo_cols].values.copy()
    n_diseases = X.shape[0]

    print("Propagating...")
    for i in range(n_diseases):
        active = np.where(X[i] > 0)[0]
        propagated = set(active)
        for h_idx in active:
            propagated.update(ancestors_idx[h_idx])
        X[i, list(propagated)] = 1
    omim_df[hpo_cols] = X

    return omim_df


def load_data2(args, hpo_graph, omim_df):

    print("Propagation ancestrale")
    omim_df = propagate_annotations(omim_df, hpo_graph)

    # =========================
    # HPO columns
    # =========================
    hpo_cols = [c for c in omim_df.columns if c.startswith("HP:")]

    # =========================
    # Disease features
    # =========================
    print("Build TF-IDF features")
    X = omim_df[hpo_cols].values.astype(np.float32)
    tfidf = TfidfTransformer(norm='l2')
    features = tfidf.fit_transform(X)

    print(features.shape)
    print(features[:][:5])

    # =========================
    # Disease graph
    # =========================
    print("Compute RFA / KNN graph")

    adj, D_high = compute_rfa(
        features,
        mode='features',
        k_neighbours=10,
        sym=True,
        connected=True,
        sigma=1.0,
        distlocal='cosine')

    adj = scipy.sparse.csr_matrix(adj)

    print(adj.shape)
    print(features.shape)
    
    # =========================
    # Split edges
    # =========================
    data = {}

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
        adj,
        args.val_prop,
        args.test_prop,
        args.split_seed)

    data['adj_train'] = adj_train

    data['train_edges'] = train_edges
    data['train_edges_false'] = train_edges_false

    data['val_edges'] = val_edges
    data['val_edges_false'] = val_edges_false

    data['test_edges'] = test_edges
    data['test_edges_false'] = test_edges_false

    data['adj_train_norm'], data['features'] = process(
        adj_train,
        features,
        args.normalize_adj,
        args.normalize_feats)
    return data