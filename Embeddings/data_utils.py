import torch
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from collections import deque
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from frechetmean import frechet_mean


def compute_depths(objects, data):
    children = {i: [] for i in range(len(objects))}
    for i in range(len(objects)):
        for parent in data.pos_neighbors[i]:
            children[parent].append(i)
    has_parent = set()
    for i in range(len(objects)):
        for parent in data.pos_neighbors[i]:
            has_parent.add(i)
    roots = [i for i in range(len(objects)) if i not in has_parent]
    print(roots)
    depths = np.full(len(objects), -1)
    queue = deque()
    for r in roots:
        depths[r] = 0
        queue.append(r)
    while queue:
        node = queue.popleft()
        for child in children[node]:
            if depths[child] == -1:
                depths[child] = depths[node] + 1
                queue.append(child)
    return depths


def f_active_terms(row, hpo_cols, node2id, deprecated):
    active=[]
    for term in hpo_cols:
        if row[term]==1:
            resolved = deprecated.get(term, term)
            if resolved in node2id:
                active.append(resolved)
    return active
    

def len_active_terms(row, hpo_cols, node2id, deprecated):
    active=[]
    for term in hpo_cols:
        if row[term]==1:
            resolved = deprecated.get(term, term)
            if resolved in node2id:
                active.append(resolved)
    return len(active)


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


def build_fuzzy_correspondence(df, threshold=0.95):
    omim = df[df['database_id'].str.startswith('OMIM:')][['disease_name', 'database_id']].drop_duplicates()
    orpha = df[df['database_id'].str.startswith('ORPHA:')][['disease_name', 'database_id']].drop_duplicates()

    # Normaliser
    omim_names  = omim['disease_name'].str.lower().str.strip()
    orpha_names = orpha['disease_name'].str.lower().str.strip()

    # TF-IDF sur les caractères (ngram sur chars capture mieux les variations médicales)
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
    all_names  = pd.concat([omim_names, orpha_names])
    vectorizer.fit(all_names)

    omim_vecs  = vectorizer.transform(omim_names)   # (n_omim, vocab)
    orpha_vecs = vectorizer.transform(orpha_names)  # (n_orpha, vocab)

    # Matrice de similarité (n_omim, n_orpha)
    sim_matrix = cosine_similarity(omim_vecs, orpha_vecs)

    # Pour chaque OMIM, prendre le meilleur match ORPHA
    best_orpha_idx   = sim_matrix.argmax(axis=1)
    best_orpha_score = sim_matrix.max(axis=1)

    results = []
    for i, (omim_row, orpha_idx, score) in enumerate(
        zip(omim.itertuples(), best_orpha_idx, best_orpha_score)
    ):
        if score >= threshold:
            orpha_row = orpha.iloc[orpha_idx]
            results.append({
                'omim_id' : omim_row.database_id,
                'orpha_id': orpha_row['database_id'],
                'omim_name' : omim_row.disease_name,
                'orpha_name': orpha_row['disease_name'],
                'similarity' : round(score, 4),
            })

    df_corr = pd.DataFrame(results).sort_values('similarity', ascending=False)
    print(f"Correspondances exactes (=1.0)  : {(df_corr['similarity'] == 1.0).sum()}")
    print(f"Correspondances >= {threshold}      : {len(df_corr)}")
    return df_corr


def find_gene_correspondence(df, disease_col, gene_col):
    gene_sets = df.groupby(disease_col)[gene_col].apply(set)
    omim_sets = gene_sets[gene_sets.index.str.startswith("OMIM")]
    orpha_sets = gene_sets[gene_sets.index.str.startswith("ORPHA")]
    matches = []
    for omim_id, omim_genes in omim_sets.items():
        for orpha_id, orpha_genes in orpha_sets.items():
            if omim_genes == orpha_genes:
                matches.append({"omim_id": omim_id, "orpha_id": orpha_id, "genes": ", ".join(omim_genes)})
    
    return pd.DataFrame(matches)


def compute_disease_barycenters(profils_omim, node2id, model, deprecated, weights=None, normalize=False, c=1):
    
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

    profils_omim['barycenter'] = barycenters
    return profils_omim
 

def compare_barycenters(profils_uniform, profils_weighted, model, node2id, deprecated, weights, disease_id=3):
    model.eval()
    W = model.weight.detach().cpu().numpy()

    row = profils_uniform.loc[disease_id]
    hpo_cols = [col for col in profils_uniform.columns if col.startswith('HP:')]
    active = [col for col in hpo_cols if row[col] == 1 and deprecated.get(col, col) in node2id]

    embs = np.stack([W[node2id[deprecated.get(t, t)]] for t in active])
    bary_uni = profils_uniform.loc[disease_id, 'barycenter']
    bary_w = profils_weighted.loc[disease_id, 'barycenter']
    root = W[node2id['HP:0000001']]

    # PCA commune pour les deux plots
    all_points = np.vstack([embs, bary_uni, bary_w, root])
    pca = PCA(n_components=2)
    proj = pca.fit_transform(all_points)
    max_norm = np.linalg.norm(proj, axis=1).max()
    proj = proj / max_norm

    embs_2d = proj[:-3]
    bary_uni2d = proj[-3]
    bary_w2d = proj[-2]
    root_2d = proj[-1]

    norms = np.linalg.norm(embs, axis=1)
    w_vals = np.array([weights.get(deprecated.get(t, t), 1e-6) for t in active])
    w_norm = w_vals / w_vals.sum()
    sizes_w = 20 + 300 * w_norm / w_norm.max()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, bary_2d, bary_arr, title, sizes in zip(
        axes[:2],
        [bary_uni2d, bary_w2d],
        [bary_uni,   bary_w],
        ['Poids uniformes', 'Poids information content'],
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

        ax.scatter(*bary_2d, s=250, c='red',  marker='*', zorder=5,
                   label=f'Barycentre (norme={np.linalg.norm(bary_arr):.3f})')
        ax.scatter(*root_2d, s=150, c='blue', marker='D', zorder=5, label='Racine')

        plt.colorbar(sc, ax=ax, label='norme HPO')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
        ax.set_title(title)

    # Panneau 3 : distribution des normes colorée par poids
    ax3 = axes[2]
    sc3 = ax3.scatter(norms, w_norm, c=norms, cmap='plasma', s=30, alpha=0.7)
    ax3.axvline(np.linalg.norm(bary_uni), color='green',  linestyle='--',
                label=f'Barycentre uniforme ({np.linalg.norm(bary_uni):.3f})')
    ax3.axvline(np.linalg.norm(bary_w),   color='red',    linestyle='--',
                label=f'Barycentre pondéré ({np.linalg.norm(bary_w):.3f})')
    ax3.set_xlabel('Norme du terme HPO (proximité du bord)')
    ax3.set_ylabel('Poids normalisé')
    ax3.set_title('Poids vs norme des termes actifs')
    ax3.legend(fontsize=8)
    plt.colorbar(sc3, ax=ax3, label='norme')

    fig.suptitle(f'Maladie {disease_id} — {len(active)} termes HPO', fontsize=13)
    plt.tight_layout()
    plt.show()


def visualize_barycenter(profils_omim, node2id, model, deprecated, disease_ids, weights=None):
    model.eval()
    W = model.weight.detach().cpu().numpy()
    hpo_cols = [col for col in profils_omim.columns if col.startswith('HP:')]
    
    diseases_dict = {}
    for d in disease_ids:
        row = profils_omim.loc[d]
        active = [col for col in hpo_cols if row[col] == 1 and deprecated.get(col, col) in node2id]
        embs = np.stack([W[node2id[deprecated.get(t, t)]] for t in active])
        bary = row['barycenter']
        diseases_dict[d]={'active': active, 'embs': embs, 'bary': bary}
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
                    arrowprops=dict(arrowstyle='-', color=color,
                                    lw=0.5, alpha=0.5))

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