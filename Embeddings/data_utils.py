import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tempfile, os

from collections import defaultdict
from pathlib import Path
from itertools import combinations
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from sklearn.decomposition import PCA

from frechetmean import frechet_mean


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


def f_active_terms(row, hpo_cols, node2id, deprecated):
    active = []
    for term in hpo_cols:
        if row[term]==1:
            resolved = deprecated.get(term, term)
            if resolved in node2id:
                active.append(resolved)
    return active
    

def len_active_terms(row, hpo_cols, node2id, deprecated):
    active = []
    for term in hpo_cols:
        if row[term] == 1:
            resolved = deprecated.get(term, term)
            if resolved in node2id:
                active.append(resolved)
    return len(active)


def find_lca(u, v, ancestors, depths):
    """Trouve le plus proche ancêtre commun entre deux noeuds"""
    if u==v:
        return u, depths[u]
    parents_u = ancestors[u]
    parents_v = ancestors[v]
    parents = set(parents_u) & set(parents_v)
    if not parents:
        return None, 0  # -np.inf ?
    lca = max(parents, key=lambda n: depths[n])
    return lca, depths[lca]


def shortest_path(d1, d2, depths, ancestors):
    '''
    Entrées :
        - d1, d2 : maladies sous la forme de liste de leurs termes HPO actifs.
        - depths : dictionnaire des profondeurs.
        - ancestors : dictionnaire des ancêtres.
    Sortie :
        - Plus court chemin moyen des termes de d1 aux termes de d2.
    Attention, shortest_path(d1, d2)
    '''
    dists=[]
    for t in d1:
        d_t = depths[t]
        for s in d2:
            _, d_lca = find_lca(t, s, ancestors, depths)
            dists.append(d_t + depths[s] - 2*d_lca)
    return np.mean(dists)


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


@torch.no_grad()
def evaluate_embedding(model, objects, edges, node2id, device):
    """
    Retourne MAP et mean rank.
    """
    model.eval()
    W = model.weight.to(device)  # (N, dim)

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
        dists = model.manifold.distance(u_emb, W, model.c)  # (N,)
        dists[u] = float('inf')                    # exclure u lui-même
        dists_np = dists.cpu().numpy()

        max_finite = dists_np[np.isfinite(dists_np)].max()
        dists_np[~np.isfinite(dists_np)] = max_finite + 1.0

        # Rang des voisins
        sorted_ind = np.argsort(dists_np)
        ranks = np.where(np.isin(sorted_ind, list(neighbors)))[0] + 1
        # Correction : soustraire les rangs des autres voisins placés avant
        n_neighbors = len(neighbors)
        corrected_ranks = ranks - np.arange(n_neighbors)
        ranks_all.extend(corrected_ranks.tolist())

        # AP
        labels.fill(0)
        labels[list(neighbors)] = 1
        ap_scores.append(average_precision_score(labels, -dists_np))

    map_score = float(np.mean(ap_scores))
    mean_rank = float(np.mean(ranks_all))

    model.train()
    return {'MAP': map_score, 'Rang moyen' : mean_rank}


def compute_disease_barycenters(profils_omim, node2id, model, colname, deprecated, weights=None, normalize=False, c=1):
    '''
    Entrées : 
        - profils_omim : dataset d'annotations de maladies.
        - node2id : dictionnaire noeud-index dans le graphe.
        - model : modèle à partir duquel charger les représentations des noeuds.
        - weights : pondérations éventuelles.
        - normalize : True si l'on normalise les poids.
    Sortie :
        - profils_omim : dataset d'annotations de maladie avec une colonne 'barycenter', donnant les
        coordonnées de la maladie dans le disque de Poincaré.
    '''
    model.eval()
    W = model.weight.detach().cpu()
    hpo_cols = [c for c in profils_omim.columns if c.startswith('HP')]

    col_meta = {}
    for col in hpo_cols:
        resolved = deprecated.get(col, col)
        if resolved in node2id:
            w = weights[resolved] if (weights is not None and resolved in weights) else 1.0
            col_meta[col]=(W[node2id[resolved]], w)  # (Coordonnée, pondération)

    valid_cols = list(col_meta.keys())
    barycenters = []
    for _, row in tqdm(profils_omim.iterrows(), total=len(profils_omim), desc="Barycentres"):
        active = [(col_meta[col][0], col_meta[col][1])
                  for col in valid_cols if row[col] == 1]  # Termes actifs

        if len(active) < 1:
            barycenters.append(None)
            continue

        points = torch.stack([a[0] for a in active])
        if weights is None:
            w = None
        else:
            w = torch.tensor([a[1] for a in active], dtype=torch.float32)
            if normalize:
                w = w / w.sum()
        barycenter = frechet_mean(points, c, w)
        barycenters.append(barycenter.numpy())

    profils_omim[colname] = barycenters
    return profils_omim
 

def compare_barycenters(df, colname1, colname2, model, node2id, deprecated, weights, disease_id=3):
    model.eval()
    W = model.weight.detach().cpu().numpy()

    row = df.loc[disease_id]
    hpo_cols = [col for col in df.columns if col.startswith('HP:')]
    active = [col for col in hpo_cols if row[col] == 1 and deprecated.get(col, col) in node2id]

    embs = np.stack([W[node2id[deprecated.get(t, t)]] for t in active])
    bary1 = df.loc[disease_id, colname1]
    bary2 = df.loc[disease_id, colname2]
    root = W[node2id['HP:0000118']]

    # PCA commune pour les deux plots
    all_points = np.vstack([embs, bary1, bary2, root])
    pca = PCA(n_components=2)
    proj = pca.fit_transform(all_points)
    max_norm = np.linalg.norm(proj, axis=1).max()
    proj = proj / max_norm

    embs_2d = proj[:-3]
    bary1_2d = proj[-3]
    bary2_2d = proj[-2]
    root_2d = proj[-1]

    norms = np.linalg.norm(embs, axis=1)
    w_vals = np.array([weights.get(deprecated.get(t, t), 1e-6) for t in active])
    w_norm = w_vals / w_vals.sum()
    sizes_w = 20 + 300 * w_norm / w_norm.max()

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for ax, bary_2d, bary_arr, title, sizes in zip(
        axes[:2],
        [bary1_2d, bary2_2d],
        [bary1,   bary2],
        [colname1, colname2],
        [np.full(len(active), 40), sizes_w]
    ):
        circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', linewidth=0.8)
        ax.add_patch(circle)

        sc = ax.scatter(embs_2d[:, 0], embs_2d[:, 1],
                        s=sizes, alpha=0.6,
                        c=norms, cmap='plasma', vmin=0, vmax=1, zorder=3)

        for pt in embs_2d:
            ax.plot([pt[0], bary_2d[0]], [pt[1], bary_2d[1]],
                    color='gray', alpha=0.15, linewidth=0.5, zorder=2)

        ax.scatter(*bary_2d, s=250, c='red',  marker='X', zorder=5,
                   label=f'Barycentre (norme={np.linalg.norm(bary_arr):.3f})')
        ax.scatter(*root_2d, s=150, c='blue', marker='X', zorder=5, label='Racine')

        plt.colorbar(sc, ax=ax, label='norme HPO')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
        ax.set_title(title)


    fig.suptitle(f'Maladie {disease_id} — {len(active)} termes HPO', fontsize=13)
    plt.tight_layout()
    plt.show()


def save_method(dict_method, method_name, savedir=Path("../data/utils")):
    savedir.mkdir(parents=True, exist_ok=True)
    entry = dict_method[method_name]  # dict de tableaux
    path = SAVE_DIR / f"{method_name}.npz"
    fd, tmp_path = tempfile.mkstemp(dir=SAVE_DIR, suffix=".npz")
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


def build_pairs_dictionary(pairs, df_omim, df_orpha, df):
    '''
    Entrées :
        - pairs : ensemble (set) de paires de maladies (i,j).
        - df_omim, df_orpha : bases de données Omim et Orpha.
        - df : base de données initiale non filtrée, d'où l'on peut tirer les noms des maladies.
    Sortie :
        - Dictionnaire qui indique pour chaque paire, le nom de la maladie associée, le nombre de
        termes actifs dans chacune des deux bases et la liste de ces termes.
    '''
    hpo_cols = [c for c in df_omim.columns if c.startswith('HP')]
    X_omim = df_omim[hpo_cols].astype(int).values
    X_orpha = df_orpha[hpo_cols].astype(int).values

    omim_to_idx = {v: i for i, v in enumerate(df_omim['database_id'].values)} 
    orpha_to_idx = {v: i for i, v in enumerate(df_orpha['database_id'].values)}

    idx2omim = {t: v for v, t in omim_to_idx.items()}
    idx2orpha = {t: v for v, t in orpha_to_idx.items()}

    dico = {}
    for (i, j) in pairs:
        active_omim = np.where(X_omim[i,:]==1)[0]
        active_orpha = np.where(X_orpha[j, :]==1)[0]
        omim_hpos = [hpo_cols[k] for k in active_omim]
        orpha_hpos = [hpo_cols[k] for k in active_orpha]
        disease_i = pd.unique(df[df['database_id']==idx2omim[i]]['disease_name'])[0]
        disease_j = pd.unique(df[df['database_id']==idx2orpha[j]]['disease_name'])[0]

        dico[(i, j)] = {
        "omim":  {"name": disease_i, "count": len(omim_hpos), "terms": omim_hpos},
        "orpha": {"name": disease_j, "count": len(orpha_hpos), "terms": orpha_hpos},
        "commun":set(omim_hpos)&set(orpha_hpos),
        }
    return dico 

# =============================================================================================
# ===================== Fonctions écrites mais non utilisées finalement =======================
# =============================================================================================
from itertools import combinations


def build_disease_correspondence(df):
    """
    Construit une table de correspondance OMIM <-> ORPHA basée sur le nom des maladies.
    """
    omim = df[df['database_id'].str.startswith('OMIM:')][['disease_name', 'database_id']].drop_duplicates()
    orpha = df[df['database_id'].str.startswith('ORPHA:')][['disease_name', 'database_id']].drop_duplicates()

    correspondence = omim.merge(orpha, on='disease_name', suffixes=('_omim', '_orpha'))
    
    return correspondence.rename(columns={
        'database_id_omim' : 'omim_id',
        'database_id_orpha': 'orpha_id'
    })[['disease_name', 'omim_id', 'orpha_id']]


def disease_hpo_distances(disease_id, X, hpo_cols, node2id, model, deprecated):
    '''
    Entrées : 
        - X : annotations sous la forme d'une matrice de taille n_maladies x n_hpo.
        - disease_id : index de la maladie dans X.
        - hpo_cols : liste des noms des colonnes de la matrice.
        - node2id : dictionnaire {noeud : index dans le graphe/la représentation}
        - model : modèle donnant accès à la représentation apprise.
    Sortie :
        - Dataframe regroupant les statistiques descriptives relatives aux distances entre les termes
        actifs de la maladie considérée dans le disque de Poincaré.
    '''
    row = X[disease_id]
    active_terms = np.where(row == 1)[0]

    if len(active_terms) < 2:
        print(f"Strictement moins de 2 termes HPO pour {disease_id}")
        return None
    model.eval()
    W = model.weight.detach().cpu().numpy()
    embs = np.array([W[node2id[hpo_cols[i]]] for i in active_terms])

    # Calculer toutes les distances par paires
    pairs = list(combinations(range(len(active_terms)), 2))
    distances = {}
    for i, j in pairs:
        u = torch.tensor(embs[i], dtype=torch.float64)
        v = torch.tensor(embs[j], dtype=torch.float64)
        d = model.manifold.distance(u, v, c=1).item()
        distances[(active_terms[i], active_terms[j])] = d

    dist_values = np.array(list(distances.values()))

    stats = {
        'disease': disease_id,
        'n_terms': len(active_terms),
        'mean_dist': dist_values.mean(),
        'max_dist': dist_values.max(),
        'min_dist': dist_values.min(),
        'std_dist': dist_values.std(),
    }

    return active_terms, embs, distances, stats


def compute_all_distances(df, model, node2id, deprecated):
    '''
    Version généralisée de la fonction précédente. Calcule les statistiques descriptives
    des distances entre termes actifs dans le disque de Poincaré pour chaque maladie de df.
    '''
    hpo_cols = [c for c in df.columns if c.startswith('HP')]
    results = []

    model.eval()
    W = model.weight.detach().cpu().numpy()

    col_to_idx = {}
    for col in hpo_cols:
        resolved = deprecated.get(col, col)
        if resolved in node2id:
            col_to_idx[col] = node2id[resolved]

    valid_cols = [col for col in hpo_cols if col in col_to_idx]
    col_ids = torch.tensor([col_to_idx[col] for col in valid_cols], dtype=torch.long)

    M = torch.tensor(df[valid_cols].values, dtype=torch.bool)
    E = W[col_ids]

    names = df.iloc[:, 0].tolist()

    for (disease_id, name), row_mask in tqdm(zip(zip(df.index, names), M),total=len(df)):
        idx = row_mask.nonzero(as_tuple=True)[0]
        if len(idx) < 2:
            continue
        embs = E[idx]
        n = len(embs)
        ii, jj = torch.triu_indices(n, n, offset=1)
        dists = model.manifold.distance(
            torch.tensor(embs[ii], dtype=torch.float32), torch.tensor(embs[jj], dtype=torch.float32), c=1.
            )
        dists = np.array(dists)
        results.append({
            'disease': disease_id,
            'name': name,
            'n_terms': n,
            'sum': dists.sum(),
            'mean_dist': dists.mean(),
            'max_dist': dists.max(),
            'min_dist': dists.min(),
            'std_dist': dists.std(),
        })

    return pd.DataFrame(results).reset_index(drop=True)


def visualize_barycenter(df, colname, node2id, model, deprecated, disease_ids, weights=None):
    model.eval()
    W = model.weight.detach().cpu().numpy()
    hpo_cols = [col for col in df.columns if col.startswith('HP:')]
    
    diseases_dict = {}
    for d in disease_ids:
        row = df.loc[d]
        active = [col for col in hpo_cols if row[col] == 1 and deprecated.get(col, col) in node2id]
        embs = np.stack([W[node2id[deprecated.get(t, t)]] for t in active])
        bary = row[colname]
        diseases_dict[d] = {'active': active, 'embs': embs, 'bary': bary}
    root = W[node2id['HP:0000001']]

    # PCA dans l'espace ambiant sur les termes actifs + barycentre + racine
    all_points = np.vstack(
        [root.reshape(1, -1)] + 
        [d['bary'].reshape(1, -1) for d in diseases_dict.values()] + 
        [d['embs'] for d in diseases_dict.values()])
    pca = PCA(n_components=2)
    projected = pca.fit_transform(all_points)

    max_norm = np.linalg.norm(projected, axis=1).max()
    projected = projected / max_norm
    root_2d = projected[0]
    idx = 1
    for d in diseases_dict:
        diseases_dict[d]['bary_2d'] = projected[idx]
        idx += 1
    for d in diseases_dict:
        n = len(diseases_dict[d]['embs'])
        diseases_dict[d]['embs_2d'] = projected[idx:idx + n]
        idx += n
        
    # --- Palette de couleurs pour distinguer les maladies ---
    palette = cm.get_cmap('tab10', len(diseases_dict))
    colors = {did: palette(i) for i, did in enumerate(diseases_dict)}

    fig, ax = plt.subplots(figsize=(9, 9))
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', linewidth=0.8, zorder=1)
    ax.add_patch(circle)
    ax.scatter(*root_2d, s=180, c='black', marker='D', zorder=6,
               label='Racine HP:0000001')

    for d, data in diseases_dict.items():
        color = colors[d]
        embs_2d = data['embs_2d']
        bary_2d = data['bary_2d']
        active = data['active']

        # Taille des points selon poids optionnels
        if weights is not None:
            sizes = np.array([
                weights.get(deprecated.get(t, t), 1e-4) for t in active
            ])
            sizes = 20 + 200 * (sizes - sizes.min()) / (
                sizes.max() - sizes.min() + 1e-10)
        else:
            sizes = np.full(len(active), 40)

        # Lignes terme - barycentre
        for pt in embs_2d:
            ax.plot([pt[0], bary_2d[0]], [pt[1], bary_2d[1]],
                    color=color, alpha=0.12, linewidth=0.5, zorder=2)

        # Termes HPO actifs
        ax.scatter(embs_2d[:, 0], embs_2d[:, 1],
                   s=sizes, color=color, alpha=0.5, zorder=3,
                   edgecolors='none')

        # Barycentre (étoile colorée + contour noir pour lisibilité)
        ax.scatter(*bary_2d, s=250, color=color, marker='*',
                   zorder=5, edgecolors='black', linewidths=0.6,
                   label=f'Maladie {d} ({len(active)} termes)')

        # Étiquette du barycentre
        ax.annotate(str(d), xy=bary_2d,
                    xytext=(bary_2d[0] + 0.03, bary_2d[1] + 0.03),
                    fontsize=8, color=color, zorder=7,
                    arrowprops=dict(arrowstyle='-', color=color, lw=0.5, alpha=0.5))

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
    ax.set_title(
        f"Barycentres de {len(diseases_dict)} maladies — projection PCA (boule de Poincaré)\n"
        f"variance expliquée : {pca.explained_variance_ratio_.sum() * 100:.1f}%"
    )
    plt.tight_layout()
    plt.show()
