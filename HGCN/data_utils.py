import torch
import numpy as np
import pandas as pd
import tempfile, os

from sklearn.metrics import average_precision_score
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm 


deprecated = {
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


def save_method(dict_method, method_name, savedir=Path("../data/utils")):
    savedir.mkdir(parents=True, exist_ok=True)
    entry = dict_method[method_name]  # dict de tableaux
    path = savedir / f"{method_name}.npz"
    fd, tmp_path = tempfile.mkstemp(dir=savedir, suffix=".npz")
    os.close(fd)
    np.savez_compressed(tmp_path, **entry)
    os.replace(tmp_path, path)


def load_all_methods(savedir=Path("../data/utils")):
    savedir.mkdir(parents=True, exist_ok=True)
    dict_method = {}
    for f in savedir.glob("*.npz"):
        loaded = np.load(f)
        dict_method[f.stem] = {k: loaded[k] for k in loaded.files}
        loaded.close()
    return dict_method
    

def get_ancestors0(G, node):
    visited = set()
    queue = list(G.successors(node))
    while queue:
        current = queue.pop()
        if current not in visited:
            visited.add(current)
            queue.extend(G.successors(current))
    return visited


def propagate_terms(df, hpo_cols, ancestors, depths, k=None):
    '''
    Pour propager les termes HPO jusqu'à k ancêtres. Compatible avec des annotations non-binaires, 
    avec les fréquences d'apparition du terme dans [0,1]
    Entrées :
        - df : dataset de maladies,
        - hpo_cols : liste des colonnes de termes HPO ([c for c in df.columns if c.startswith('HP')]),
        - ancestors : dictionnaire des ancêtres des termes HPO dans l'ontologie,
        - depths : dictionnaire des profondeurs dans l'ontologie,
        - k : profondeur jusqu'où on souhaite propager.
    Sortie :
        - df_out : dataset de maladies avec propagation ancestrale jusqu'au niveau k.
    '''
    mat = df[hpo_cols].to_numpy().copy()
    col2idx = {c: i for i, c in enumerate(hpo_cols)}
    for row in mat:
        active_terms = [hpo_cols[i] for i, v in enumerate(row) if v > 0]
        for term in active_terms:
            for anc in ancestors.get(term, []):
                if anc not in col2idx:
                    continue
                if k is None or depths[term] - depths[anc] <= k:
                    row[col2idx[anc]] = min(row[col2idx[anc]]+row[col2idx[term]], 1)  # Au cas où plusieurs termes actifs auraient le même parent.
    df_out = df.copy()
    df_out[hpo_cols] = mat
    return df_out


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
    

def f_ground_truth(work_omim, work_orpha, df_orpha_omim):
    omim_to_idx = {v: i for i, v in enumerate(work_omim['database_id'].values)} 
    orpha_to_idx = {v: i for i, v in enumerate(work_orpha['database_id'].values)}

    ground_truth = []
    valid_omim = []
    valid_orpha = []
    for _, row in df_orpha_omim.iterrows():
        if row['omim_id'] in omim_to_idx and row['orpha_id'] in orpha_to_idx:
            i = omim_to_idx[row['omim_id']]
            j = orpha_to_idx[row['orpha_id']]
            ground_truth.append((i, j))
            valid_omim.append(row['omim_id'])
            valid_orpha.append(row['orpha_id'])
    gt_set = set(ground_truth)
    return gt_set, valid_omim, valid_orpha


def precompute(df, node2id):
    '''
    Renvoie pour chaque maladie (ligne) du dataframe df la liste des termes HPO actifs.
    '''
    hpo_cols = [c for c in df.columns if c.startswith('HP')]
    X = df[hpo_cols].to_numpy(dtype=bool)
    resolved_cols = np.array(
        [deprecated.get(col, col) if deprecated.get(col, col) in node2id else None for col in hpo_cols], 
        dtype=object)
    valid_mask = resolved_cols != np.array(None)
    X_valid = X[:, valid_mask]
    resolved_valid = resolved_cols[valid_mask]
    terms = [list(resolved_valid[row_mask]) for row_mask in X_valid]
    return terms

def precompute_weights(active, ic):
    a = np.zeros(len(active))
    for j in range(len(active)):
        terms = active[j]
        ic_j = np.array([ic[t] for t in terms])
        a[j] = np.mean(ic_j)
    return a/a.sum()


def compute_information_content(df_omim, G_hpo, deprecated=deprecated):
    '''
    Entrées :
        - df_omim : dataset de maladies annotées,
        - G_hpo : graphe de l'ontologie HPO,
        - deprecated : liste de termes apparaissant dans les maladies et mis à jour depuis.
    Sortie :
        - Dictionnaire qui associe à chaque terme HPO apparaissant dans au moins une maladie la 
        valeur de l'IC.
        - diseases : dictionnaire {terme HPO : liste des maladies qui le contiennent}
        - all_diseases : dictionnaire {terme HPO : maladies qui le contiennent avec propagation ancestrale}
    NB : On pondère sur l'ensemble des noeuds du graphe, pas juste sur les maladies. Du point de vue
    de l'interprétation c'est moins élégant, car on n'a pas de probabilité d'apparition dans une maladie,
    mais au niveau des résultats c'est un peu mieux. 
    '''
    colnames = [c for c in df_omim.columns if c.startswith('HP:')]
    hp_matrix = df_omim[colnames].values
    ids = df_omim.index.tolist()

    weights = defaultdict(float)
    diseases = defaultdict(set)
    all_diseases = defaultdict(set)
    ancestors = {}

    def get_ancestors(term):
        if term not in ancestors:
            ancestors[term] = get_ancestors0(G_hpo, term)
        return ancestors[term]

    row_idxs, col_idxs = np.where(hp_matrix > 0)
    for row_idx, col_idx in zip(row_idxs, col_idxs):
        disease_id = ids[row_idx]
        term = colnames[col_idx]  # terme HPO
        resolved = deprecated.get(term, term)
        
        weights[resolved] += hp_matrix[row_idx, col_idx]
        diseases[resolved].add(disease_id)
        all_diseases[resolved].add(disease_id)
        
        for ancestor in get_ancestors(resolved):
            weights[ancestor] += hp_matrix[row_idx, col_idx]
            all_diseases[ancestor].add(disease_id)

    total = sum(weights.values())
    return {t: w / total for t, w in weights.items()}, diseases, all_diseases


@torch.no_grad()
def evaluate_embeddings(embeddings, G_hpo, objects, node2id, manifold):
    """
    Retourne le MAP et le rang moyen.
    """
    edges = np.array([(node2id[u], node2id[v]) for u, v in G_hpo.edges()],dtype=np.int32)
    W = torch.from_numpy(embeddings.copy())

    pos_neighbors = defaultdict(set)
    for u, v in edges:
        pos_neighbors[int(u)].add(int(v))

    ap_scores = []
    ranks_all = []
    N = W.shape[0]
    labels = np.zeros(N)

    for obj in tqdm(objects):
        u = int(node2id[obj])
        neighbors = pos_neighbors.get(u, set())
        if not neighbors:
            continue

        # Distances GPU avec la bonne courbure
        u_emb = W[u].unsqueeze(0).expand(N, -1)   # (N, dim)
        dists = np.sqrt(manifold.sqdist(u_emb, W, 1.).numpy())  # (N,)
        dists[u] = float('inf')                    # exclure u lui-même

        max_finite = dists[np.isfinite(dists)].max()
        dists[~np.isfinite(dists)] = max_finite + 1.0

        # Rang des voisins
        sorted_ind = np.argsort(dists)
        ranks = np.where(np.isin(sorted_ind, list(neighbors)))[0] + 1
        # Correction : soustraire les rangs des autres voisins placés avant
        n_neighbors = len(neighbors)
        corrected_ranks = ranks - np.arange(n_neighbors)
        ranks_all.extend(corrected_ranks.tolist())

        # AP
        labels.fill(0)
        labels[list(neighbors)] = 1
        ap_scores.append(average_precision_score(labels, -dists))

    map_score = float(np.mean(ap_scores))
    mean_rank = float(np.mean(ranks_all))

    return map_score, mean_rank
