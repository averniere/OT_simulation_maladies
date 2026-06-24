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
from itertools import combinations
from tqdm import tqdm 

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


def load_data(args, hpo_graph, omim_df, ancestors, depths, p=0, cache_dir=".cache"):

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
    hpo_cols = [c for c in omim_df.columns if c.startswith('HP')]
    if p>0:
        print("Propagate HPO annotations")
        omim_df_w = propagate_terms(omim_df, hpo_cols, ancestors, depths, p)
    else:
        omim_df_w = omim_df.copy()
    all_hpo_in_graph = [n for n in nodes if n in omim_df_w.columns]
    missing_hpo = [n for n in nodes if n not in omim_df_w.columns]

    print(f"Termes HPO avec colonne dans omim_df : {len(all_hpo_in_graph)}")
    print(f"Termes HPO sans colonne (nœuds internes purs) : {len(missing_hpo)}")

    hpo_cols_w = [col for col in omim_df_w.columns if col in node2idx]
    # transformer = TfidfTransformer()
    # hpo_matrix = transformer.fit_transform(omim_df[hpo_cols].values)
    # omim_df[hpo_cols] = hpo_matrix.toarray()
    print("Load features")
    
    omim_features = np.zeros((len(nodes), len(omim_df_w)))  # Matrice binaire avec un 1 si un noeud est présent dans une maladie
    for col in all_hpo_in_graph:
        idx = node2idx[col]
        omim_features[idx] = omim_df[col].values.astype(np.float32)  # hpo_matrix[:, hpo_cols.index(col)].toarray().ravel()
    # Propagagtion des maladies si nécessaire vers les parents
    for node in nx.topological_sort(hpo_graph):  # du bas vers le haut
        idx = node2idx[node]
        if omim_features[idx].sum() == 0:  # Si pas de maladie associée
            children = list(hpo_graph.predecessors(node))  # Parents ?
            if children:
                child_feats = np.array([omim_features[node2idx[c]] for c in children])
                if child_feats.sum() > 0:
                    omim_features[idx] = child_feats.mean(axis=0)

    print("Compute structural features")
    features = compute_structural_features(hpo_graph, nodes, node2idx, omim_features)
    features = scipy.sparse.csr_matrix(features)

    zero_nodes = (omim_features.sum(axis=1) == 0).sum()
    print(f"Nœuds avec features nulles : {zero_nodes} / {len(nodes)}")

    #omim_reindexed = omim_df[sorted(hpo_cols, key=lambda x: node2idx[x])]
    print(adj.shape)       # doit être (n_terms, n_terms)
    print(features.shape)  # doit être (n_terms, n_diseases)
    
    data = {'adj_train': adj, 'features': features}
    print("Load train, test and validation datasets")
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                    adj, args.val_prop, args.test_prop, args.split_seed, hpo_graph, node2idx)
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
    
    root = [n for n, d in hpo_graph.out_degree() if d == 0]
    if root:
        depths = nx.single_source_shortest_path_length(hpo_graph.reverse(), root[0])
        depth = np.array([depths.get(n, 0) for n in nodes]).reshape(-1, 1)
    else:
        depth = np.zeros((n, 1))
    
    n_descendants = np.array([len(nx.ancestors(hpo_graph, node)) for node in nodes]).reshape(-1, 1)
    
    struct_features = np.hstack([degree, depth, n_descendants])
    struct_features = struct_features / (struct_features.max(axis=0) + 1e-8)

    if not isinstance(omim_features, np.ndarray):
        omim_features = np.array(omim_features)
    
    features = np.hstack([omim_feat, struct_features])
    return features


def get_transitive_edges(hpo_graph, node2idx):
    
    pos_edges = []
    for node in hpo_graph.nodes():
        descendants = nx.ancestors(hpo_graph, node)
        for desc in descendants:
            pos_edges.append([
                node2idx[node],
                node2idx[desc]
            ])

    return np.array(pos_edges)


def mask_edges(adj, val_prop, test_prop, seed, hpo_graph, node2idx):
    # Positive edges (voisins)
    print("Load positive edges")
    np.random.seed(seed)
    # x, y = scipy.sparse.triu(adj).nonzero()
    # pos_edges = np.array(list(zip(x, y)))
    pos_edges = get_transitive_edges(hpo_graph, node2idx)
    np.random.shuffle(pos_edges)

    # Negative edges (non-voisins)
    print("Load negative edges")
    def sample_neg_edges(adj, n_samples, seed_offset, min_dist=5):
        np.random.seed(seed + seed_offset)
        n = adj.shape[0]
        undirected = nx.from_scipy_sparse_array(adj).to_undirected()
        neg_edges = []
        existing = set(zip(*adj.nonzero()))
        while len(neg_edges) < n_samples:
            i, j = np.random.randint(0, n, 2)
            if i >= j:
                continue
            if (i, j) in existing:
                continue
            try:
                d = nx.shortest_path_length(undirected, i, j)
                if d >= min_dist:
                    neg_edges.append([i, j])
            except:
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
    adj_train = adj_train  # +adj_train.T
    return (adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(test_edges_false),)


def process(adj, features, normalize_adj, normalize_feats):
    if scipy.sparse.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    zero_rows = (features.sum(dim=1) == 0)
    print(f"Noeuds avec features nulles : {zero_rows.sum().item()}")
    features[zero_rows] = torch.randn(zero_rows.sum(), features.shape[1]) * 1e-2
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


def propagate_terms(df, hpo_cols, ancestors, depths, k=None):
    mat = df[hpo_cols].to_numpy().copy()
    col2idx = {c: i for i, c in enumerate(hpo_cols)}
    for row in mat:
        active_terms = [hpo_cols[i] for i, v in enumerate(row) if v == 1]
        for term in active_terms:
            for anc in ancestors.get(term, []):
                if anc not in col2idx:
                    continue
                if k is None or depths[term] - depths[anc] <= k:
                    row[col2idx[anc]] = 1
    df_out = df.copy()
    df_out[hpo_cols] = mat
    return df_out


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


def add_edges(diseases, G, depths):
    """
    Si pi et pj sont deux termes actifs d'une même maladie, non reliés dans l'ontologie,
    ajouter une arête entre eux.
    """
    G_hpo = G.copy()
    hpo_cols = [c for c in diseases.columns if c.startswith('HP')]
    cols2id = {i: hp for i, hp in enumerate(hpo_cols)}
    X = diseases[hpo_cols].values
    existing_edges = set(G_hpo.edges())
    new_edges = set()
    for i in tqdm(range(len(diseases))):
        active_i = np.where(X[i] == 1)[0]   # termes actifs de la maladie i
        if len(active_i) < 2:
            continue
        for idx1, idx2 in combinations(active_i, 2):
            hp1, hp2 = cols2id[idx1], cols2id[idx2]
            d1, d2 = depths[hp1], depths[hp2]
            if d2 >= d1:
                edge = (hp2, hp1)
            else:
                edge = (hp1, hp2)
            if edge not in existing_edges:
                new_edges.add(edge)
    G_hpo.add_edges_from(new_edges)
    return G_hpo


def add_corresponding_terms(df1, df2, correspondances):
    """
    Pour les maladies de df1 qui ont une maladie correspondante dans df2, ajouter les termes actifs
    de df2 qui ne sont pas dans df1.
    Retourne un dataframe result avec les lignes des maladies de df1 complétées avec les termes de df2.
    """
    result = df1.copy()
    hpo_cols = [c for c in df1.columns if c.startswith('HP')]
    df1_to_idx = {v: i for i, v in enumerate(df1['database_id'].values)} 
    df2_to_idx = {v: i for i, v in enumerate(df2['database_id'].values)}
    mask = (
        correspondances['omim_id'].isin(df1_to_idx) & correspondances['orpha_id'].isin(df2_to_idx)
        )
    valid = correspondances[mask]

    hpo1 = result[hpo_cols].fillna(0).astype(int).values.copy()
    hpo2 = df2[hpo_cols].fillna(0).astype(int).values.copy()
    col_positions = result.columns.get_indexer(hpo_cols)

    for _, row in tqdm(valid.iterrows(), total=len(valid), desc="Fusion HPO"):
        d1 = row['omim_id']
        d2 = row['orpha_id']
        i1 = df1_to_idx[d1]
        i2 = df2_to_idx[d2]
        hpo1[i1] |= hpo2[i2]
    result.iloc[:, col_positions] = hpo1
    return result


deprecated={
    'HP:0006887':'HP:0001249',
    'HP:0002275':'HP:0002311',
    'HP:0002370':'HP:0002311',
    'HP:0002438':'HP:0001317',
    'HP:0004059':'HP:0006433',
    'HP:0005365':'HP:0010976',
    'HP:0005435':'HP:0011840',
    'HP:0005807':'HP:0009881',
    'HP:0007543':'HP:0000962',
    'HP:0007680':'HP:0007894',
    'HP:0007850':'HP:0030666',
    'HP:0007898':'HP:0012231',
    'HP:0009062':'HP:0008936',
    'HP:0010064':'HP:0010091',
    'HP:0012178':'HP:0012177',
    'HP:0030050':'HP:0002524',  # Suspect
    'HP:0031014':'HP:0031632',
    'HP:0100786':'HP:0001262',
    'HP:0200065':'HP:0000533'
}


def get_ancestors0(G, node):
    visited = set()
    queue = list(G.successors(node))
    while queue:
        current = queue.pop()
        if current not in visited:
            visited.add(current)
            queue.extend(G.successors(current))
    return visited