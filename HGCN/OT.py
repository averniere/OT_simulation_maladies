import numpy as np
import torch
import ot
import data
import matplotlib.pyplot as plt
import seaborn as sns

import frechetmean as fm

from joblib import Parallel, delayed
from tqdm import tqdm
from collections import defaultdict
from numpy.linalg import norm


# ===========================================================================================
# ================================== Matrices de coût =======================================
# ===========================================================================================

def compute_cost_matrix(omim, orpha, manifold, colname='barycenter'):
    """ 
    Distance de Poincaré entre barycentres.
    """
    omim_bary = torch.tensor(np.stack(omim[colname].values),  dtype=torch.float32)
    orpha_bary = torch.tensor(np.stack(orpha[colname].values), dtype=torch.float32)

    n, m = omim_bary.shape[0], orpha_bary.shape[0]

    u = omim_bary.unsqueeze(1).expand(n, m, -1).reshape(n * m, -1)
    v = orpha_bary.unsqueeze(0).expand(n, m, -1).reshape(n * m, -1)

    dists = manifold.distance(u, v, c=1)  # (n*m,)

    return dists.reshape(n, m).numpy()


def emb_norms(df_omim, df_orpha, node2id_w, model, manifold):
    hpo_cols = [c for c in df_omim.columns if c.startswith('HP:')]
    all_hpo = list(hpo_cols)
    model.eval()
    W = model.weight.detach().cpu().numpy()
    indices = [node2id_w[hpo] for hpo in all_hpo if hpo in node2id_w]
    known_pos = [i for i, hpo in enumerate(all_hpo) if hpo in node2id_w]
    W_known = torch.tensor(W[indices], dtype=torch.float32)
    origin = torch.zeros_like(W_known)
    with torch.no_grad():
        hyp_norms = manifold.distance(W_known, origin, c=1.).cpu().numpy()
    norms = np.zeros(len(all_hpo))
    norms[known_pos] = hyp_norms

    return norms, all_hpo


def compute_cost_matrix_pseudo_jacc(df_omim, df_orpha, node2id_w, model, block_size=256):
    """Hamming en pondérant par les embeddings."""
    n = df_omim.shape[0]
    m = df_orpha.shape[0]
    C = np.zeros((n, m))
    print("Compute norms")
    norms, all_hpo = emb_norms(df_omim, df_orpha, node2id_w, model)
    print("Norms computed !")
    
    A = df_omim.reindex(columns=all_hpo, fill_value=0)[all_hpo].values.astype(np.float32)
    B = df_orpha.reindex(columns=all_hpo,  fill_value=0)[all_hpo].values.astype(np.float32)

    Aw = A * norms
    Bw = B * norms

    C = Aw.sum(axis=1)[:, None] + Bw.sum(axis=1)[None, :] - 2 * (A @ Bw.T)
    # Check
    #for i, j in [(0, 0), (3, 7), (9, 14)]:
        #ref = np.dot(np.abs(A[i, :] - B[j, :]), norms)
        #new = C[i, j]
        #print(f"C[{i},{j}]  ref={ref:.6f}  new={new:.6f}  diff={abs(ref-new):.2e}")
    return C


def compute_costs_matrix_wasserstein2(
    df_omim, df_orpha, 
    node2id_w, 
    embeddings, 
    manifold,
    c,
    weights=None,
    deprecated=data.deprecated,
    S=None,
    gromov=False
    ):
    n = len(df_omim)
    m = len(df_orpha)
    hpo_cols = [c for c in df_omim.columns if c.startswith('HP:')]

    print("Precompute...")
    def precompute(df, weights=weights):
        '''
        Renvoie pour chaque maladie (ligne) du dataframe df la liste des termes HPO actifs et 
        le vecteur de poids uniformes associés.
        '''
        X = df[hpo_cols].to_numpy(dtype=bool)
        resolved_cols = np.array(
            [deprecated.get(col, col) if deprecated.get(col, col) in node2id_w else None for col in hpo_cols], 
            dtype=object)
        valid_mask = resolved_cols != np.array(None)
        X_valid = X[:, valid_mask]
        resolved_valid = resolved_cols[valid_mask]
        terms = [list(resolved_valid[row_mask]) for row_mask in X_valid]
        if weights is None: 
            w = [np.ones(len(t)) / len(t) if t else np.array([]) for t in terms]
        else : 
            w = [np.array([weights[t] for t in term]/np.sum([weights[t] for t in term])) for term in terms]
        return terms, w

    terms_i, weights_i = precompute(df_omim)  # Termes actifs, poids pour les maladies sources
    terms_j, weights_j = precompute(df_orpha)  # Termes actifs, poids pour les maladies destinations
    print("Finished !")

    all_terms = list({h for ts in terms_i + terms_j for h in ts})  # Tous les termes actifs
    term2idx = {h: k for k, h in enumerate(all_terms)}
    hpo_indices = [node2id_w[h] for h in all_terms]  # Indices selon node2id_w
    E = embeddings[hpo_indices]  # Fonctionne si embeddings est construit de la même manière que node2id_w
    if isinstance(E, np.ndarray):
        E = torch.tensor(E, dtype=torch.float32)

    idx_i = [[term2idx[h] for h in ts] for ts in terms_i]  # Index des termes actifs par maladies sources
    idx_j = [[term2idx[h] for h in ts] for ts in terms_j]  # Index des termes actifs par maladies destinations

    C = np.zeros((n, m))

    print("Precomputing full HPO distance matrix...")
    K = E.shape[0]
    D_full = np.zeros((K, K), dtype=np.float32)
    BLOCK = 128  # Réduire si encore OOM (128, 64...)
    for i in tqdm(range(0, K, BLOCK), desc="Distance matrix rows"):
        Ei = E[i:i+BLOCK]          # (b, dim)
        b = Ei.shape[0]
        for j in range(0, K, BLOCK):
            Ej = E[j:j+BLOCK]      # (b2, dim)
            b2 = Ej.shape[0]
                
            Ei_exp = Ei.unsqueeze(1).expand(b, b2, -1).reshape(b * b2, -1)
            Ej_exp = Ej.unsqueeze(0).expand(b, b2, -1).reshape(b * b2, -1)
                
            d = np.sqrt(manifold.sqdist(Ei_exp, Ej_exp, c))
            D_full[i:i+BLOCK, j:j+BLOCK] = d.reshape(b, b2).cpu().numpy()
    
    print(f"HPO distance matrix: {D_full.shape}")
    if S is not None: 
        D_full/= D_full.max()
        simi = S[np.ix_(hpo_indices, hpo_indices)]
        D_full -= D_full * simi  # éventuellement : alpha*simi

    def compute_row(i):
        if not idx_i[i]:
            return i, np.zeros(len(terms_j))
        # Ei = E[idx_i[i]]
        row = np.zeros(len(terms_j))
        valid_js = [j for j in range(len(terms_j)) if idx_j[j]]
        for j in valid_js:
            if gromov: 
                C1 = D_full[np.ix_(idx_i[i], idx_i[i])]
                C2 = D_full[np.ix_(idx_j[j], idx_j[j])]
                gm_plan = ot.gromov_wasserstein2(C1, C2)
                row[j] = np.sqrt(gm_plan)/2
            M = D_full[np.ix_(idx_i[i], idx_j[j])]
            _, row[j] = compute_transport(M, weights_i[i], weights_j[j])
        return i, row
    
    results = Parallel(n_jobs=-1)(
       delayed(compute_row)(i) for i in tqdm(range(len(df_omim)), desc="OMIM"))
       
    for i, row in results:
        C[i] = row 
    return C


# ===========================================================================================
# =============================== Méthodes de transport =====================================
# ===========================================================================================


def compute_transport(
    C: np.ndarray,
    a: np.ndarray,
    b: np.ndarray):
    n = C.shape[0]
    m = C.shape[1]
    if a is None:
        a = np.ones(n)/n
    if b is None:
        b = np.ones(m)/m
    optimal_plan = ot.emd(a, b, C, numItermax=10e6)
    optimal_cost = np.sum(optimal_plan*C)
    return optimal_plan, optimal_cost


def compute_transport_sinkhorn(
    C: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    epsilon: float,
    max_iters: int = 100000,
    tau: float = 1e-4,
    verbose: bool = False,
    ):
    n = C.shape[0]
    m = C.shape[1]
    if a is None:
        a = np.ones(n)/n
    if b is None:
        b = np.ones(m)/m
    assert np.isclose(a.sum(), 1.0), f"somme a = {a.sum()}"
    assert np.isclose(b.sum(), 1.0), f"somme b = {b.sum()}"
    optimal_plan_sinkhorn = ot.sinkhorn(a, b, C, epsilon, numItermax=max_iters, stopThr=tau)
    optimal_cost_sinkhorn = np.sum(optimal_plan_sinkhorn*C)

    if verbose:
        print(f"entropic optimal transport plan: \n{optimal_plan_sinkhorn}")
        print(f"entropic transport cost: {optimal_cost_sinkhorn}")

    return optimal_plan_sinkhorn, optimal_cost_sinkhorn


def compute_transport_sinkhorn_batch(Ms, As, Bs, epsilon, n_iter=200, device="cuda"):
    Ms = Ms.to(device)
    As = As.to(device)
    Bs = Bs.to(device)
    K = torch.exp(-Ms / epsilon)
    u = torch.ones_like(As)
    v = torch.ones_like(Bs)
    for _ in range(n_iter):
        u = As / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-9)
        v = Bs / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-9)
    P = u.unsqueeze(-1) * K * v.unsqueeze(1)
    cost = (P * Ms).sum(dim=(1, 2))
    return cost.cpu().numpy()


def compute_unbalanced(C: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    epsilon: float,
    rega,
    regb,
    max_iters: int = 100000,
    tau: float = 1e-4,
    verbose: bool = False,
    log: bool = False):
    '''
    Résout le problème de transport avec des contraintes relâchées.
    Entrées : 
        - a, b : contraintes de capacités,
        - epsilon : régularisation entropique, 
        - rega, regb : définies dans [0,1]. Si égales à 1, alors équivalent au problème de transport
        classique. Si égales à 0, alors équivalent à un problème sans contraintes de capacités.
    '''
    n = C.shape[0]
    m = C.shape[1]
    if a is None:
        a = np.ones(n)/n
    if b is None:
        b = np.ones(m)/m
    assert np.isclose(a.sum(), 1.0), f"somme a = {a.sum()}"
    assert np.isclose(b.sum(), 1.0), f"somme b = {b.sum()}"
    regm = (rega*epsilon/(1-rega) if rega<1 else np.inf, regb*epsilon/(1-regb) if regb<1 else np.inf)
    optimal_plan_sinkhorn = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, C, epsilon, regm, numItermax=max_iters, stopThr=tau)
    optimal_cost_sinkhorn = np.sum(optimal_plan_sinkhorn*C)

    if verbose:
        print(f"entropic optimal transport plan: \n{optimal_plan_sinkhorn}")
        print(f"entropic transport cost: {optimal_cost_sinkhorn}")

    return optimal_plan_sinkhorn, optimal_cost_sinkhorn


# ===========================================================================================
# ============================== Evaluation des résultats ===================================
# ===========================================================================================


def evaluate_transport(P, gt_set, C, exact=True, top_k=(1, 3, 5), verbose=True):
    """
    Évalue le plan de transport P contre la vérité terrain.
    Inputs : 
        - P : plan de transport ;
        - gt_set : correspondances exactes entre les maladies des deux bases de données sous la forme
        {(i_1,j_1), (i_2, j_2)...} ;
        - C : matrice de coût ;
        - top_k : précision, j_true est au plus la k-ième destination recevant le plus de masse.
    """
    results1 = {k: 0 for k in top_k}  # Bonnes associations par lignes
    results2 = {k: 0 for k in top_k}  # Bonnes associations par colonnes
    both = {k : 0 for k in top_k}  # Bonnes associations par ligne et par colonne
    pairs1 = {}
    pairs2 = {}
    ranks_orpha = []
    ranks_omim = []
    marginal_a = np.sum(P, axis=1)
    marginal_b = np.sum(P, axis=0)
    if exact: 
        for (i, j_true) in gt_set:
            # Colonnes triées par masse décroissante pour la ligne i
            ranked_cols = np.argsort(P[i])[::-1]
            rank = np.where(ranked_cols == j_true)[0]
            if len(rank) == 0:
                continue
            rank = rank[0] + 1
            ranks_orpha.append(rank) # Rang de la vraie maladie j_true dans la matrice de transport

            # Lignes triées par masse décroissante pour la colonne j_true
            ranked_lines = np.argsort(P[:,j_true])[::-1]
            rank2 = np.where(ranked_lines == i)[0]
            if len(rank2) == 0:
                continue
            rank2 = rank2[0] + 1
            ranks_omim.append(rank2)

            for k in top_k:
                if rank <= k and rank2 <= k:
                    results1[k] += 1
                    results2[k] += 1
                    both[k] += 1
                    if C is not None and (i, j_true) not in pairs1.keys():
                        pairs1[(i, j_true)]=[k, C[i, j_true], P[i, j_true]/marginal_a[i]]
                    if C is not None and (i, j_true) not in pairs2.keys():
                        pairs2[(i, j_true)]=[k, C[i, j_true], P[i, j_true]/marginal_b[j_true]]
                elif rank > k or rank2 > k:
                    if rank <= k:
                        results1[k] +=1
                        if C is not None and (i, j_true) not in pairs1.keys():
                            pairs1[(i, j_true)]=[k, C[i, j_true], P[i, j_true]/marginal_a[i]]
                        if C is not None and (i, j_true) not in pairs2.keys():
                            pairs2[(i, j_true)]=[0, C[i, j_true], P[i, j_true]/marginal_b[j_true]]
                    elif rank2 <= k:
                        results2[k] +=1
                        if C is not None and (i, j_true) not in pairs1.keys():
                            pairs1[(i, j_true)]=[0, C[i, j_true], P[i, j_true]/marginal_a[i]]
                        if C is not None and (i, j_true) not in pairs2.keys():
                            pairs2[(i, j_true)]=[k, C[i, j_true], P[i, j_true]/marginal_b[j_true]]
                    else:
                        if C is not None and (i, j_true) not in pairs1.keys():
                            pairs1[(i, j_true)]=[0, C[i, j_true], P[i, j_true]/marginal_a[i]]
                        if C is not None and (i, j_true) not in pairs2.keys():
                            pairs2[(i, j_true)]=[0, C[i, j_true], P[i, j_true]/marginal_b[j_true]]
        n = len(gt_set)
        if verbose:
            print(f"Paires évaluées : {n}")
            for val in ["Lignes", "Colonnes", "Lignes et Colonnes"]:  # Idéalement, créer un dico méthode --> résultat
                print(f"=========== {val} ===========")
                for k in top_k:
                    if val == "Lignes":
                        print(f"Top-{k} accuracy : {results1[k]/n:.3f} ({results1[k]}/{n})")
                    elif val == "Colonnes":
                        print(f"Top-{k} accuracy : {results2[k]/n:.3f} ({results2[k]}/{n})")
                    else:
                        print(f"Top-{k} accuracy : {both[k]/n:.3f} ({both[k]}/{n})")
            print(f" Rang moyen des maladies Orpha : {np.mean(ranks_orpha):.2f}")
            print(f" Rang moyen des maladies Omim : {np.mean(ranks_omim):.2f}")
    else:  # Pour le cas où plusieurs maladies peuvent être associées à une même maladie (à modifier)
        ranks = []
        pairs = {}
        results = {k: 0 for k in top_k}
        marginal = np.sum(P, axis=1)
        for i in gt_set.keys():
            ranked_cols = np.argsort(P[i])[::-1]
            js = gt_set[i]
            rank_j = []
            for j in js:
                rank = np.where(ranked_cols == j)[0]
                if len(rank) == 0:
                    continue
                rank = rank[0] + 1
                rank_j.append(rank)
                for k in top_k:
                    if rank <= k:
                        results[k] += 1
                        if C is not None and (i, j) not in pairs.keys():
                            pairs[(i, j)]=[k, C[i, j], P[i, j]/marginal[i]]
                if C is not None and (i, j) not in pairs.keys():
                    pairs[(i, j)]=[0, C[i, j], P[i, j]/marginal[i]]
            ranks.append(rank_j) # Rang de la vraie maladie j_true dans la matrice de transport
              
        n = len(gt_set)
        if verbose:
            print(f"Paires évaluées : {n}")
            for k in top_k:
                print(f"Top-{k} accuracy : {results[k]/n:.3f} ({results[k]}/{n})")
            #print(f" Rang moyen: {np.mean(ranks):.2f}")

    return ranks_orpha, ranks_omim, pairs1, pairs2, both


def evaluate_transport_proba(P, gt_set, seuil):
    '''
    Inspiré de https://arxiv.org/pdf/2505.24759.
    Calcule la probabilité d'association pour chaque paire de maladies (P_ij/Pj+Pij/Pi)/2.
    Seuille la matrice de probabilités ainsi obtenue. 
    Renvoie une mesure de précision (% de paires retrouvées parmi les positifs) et de rappel
    (% de paires retrouvées parmi la vérité de terrain)
    '''
    marginal_a = np.sum(P, axis=1)
    marginal_b = np.sum(P, axis=0)
    S = (np.divide(P, marginal_a[:, None])+np.divide(P, marginal_b[None, :]))/2
    S = (S > seuil).astype(int)
    predicted_positives = S.sum()
    if predicted_positives == 0:
        return None, None
    else: 
        tp=0
        for (i,j) in gt_set:
            tp += S[i,j]
        precision = tp / predicted_positives
        recall = tp/len(gt_set)
        return precision, recall


def plot_consistency(ax, reg_strengths, plan_diff, distance_diff, alpha):
    ax[0].loglog(reg_strengths, plan_diff, lw=4)
    ax[0].set_ylabel('$||P^* - P_\epsilon^*||_F$', fontsize=25)
    ax[1].tick_params(which='both', size=20)
    ax[0].grid(ls='--')
    ax[1].loglog(reg_strengths, distance_diff, lw=4)
    ax[1].axhline(alpha, color='red', linestyle='--', linewidth=1.5, label='5% seuil')
    ax[1].legend()
    ax[1].set_xlabel('Regularization Strength $\epsilon$', fontsize=25)
    ax[1].set_ylabel(r'$ 100 \cdot \frac{\langle C, P^*_\epsilon \rangle - \langle C, P^* \rangle}{\langle C, P^* \rangle} $', fontsize=25)
    ax[1].tick_params(which='both', size=20)
    ax[1].grid(ls='--') 


def plot_regularization(C, grid, alpha=5, a=None, b=None):
    plan_diff = []
    distance_diff = []
    ot_plan_sinkhorn, ot_cost_sinkhorn = compute_transport(C, a, b)

    for epsilon_prime in grid:
        epsilon = epsilon_prime * np.mean(C)
        ot_plan_sinkhorn_croissant, ot_cost_sinkhorn_croissant = compute_transport_sinkhorn(C, a, b, epsilon, 10000, 1e-4, False)

        assert ot_cost_sinkhorn_croissant != np.nan, (
            "Optimal cost is nan due to numerical instabilities."
            )
        p = norm(ot_plan_sinkhorn_croissant - ot_plan_sinkhorn)
        plan_diff.append(p)
        dist = 100 * (ot_cost_sinkhorn_croissant - ot_cost_sinkhorn)/ot_cost_sinkhorn
        distance_diff.append(dist)

    fig, ax = plt.subplots(2, 1, figsize=(16, 5*2))
    reg_strengths = np.mean(C) * grid
    plot_consistency(ax, reg_strengths, plan_diff, distance_diff, alpha)

    plt.tight_layout()
    plt.show()


def plot_unbalanced(taus, C, epsilon, gt_set):
    '''
    Entrées :
        - taus : liste de valeurs de tau à tester (tau dans [0,1]).
        - C : matrice de coût précalculée.
        - epsilon : paramètre de régularisation entropique.
        - gt_set : vérité de terrain.
    Sortie : 
        - Evolution des métriques Top 1 et Top 3, calculées pour lignes et colonnes à la fois,
        en fonction de tau, pour chaque scénario considéré (unbalanced, semi-balanced).
    '''
    scenarios = {
        'unbalanced' : lambda tau: (tau, tau), 
        'semi-a': lambda tau: (tau, 1.), 
        'semi-b': lambda tau: (1., tau)
        }
    res = {}
    for sname, sfun in tqdm(scenarios.items()):
        res[sname]={'Top 1':[], 'Top 3':[]}
        for reg in taus:
            rega, regb = sfun(reg)
            ot_plan_reg, _ = compute_unbalanced(C, None, None, epsilon, rega, regb, 10000, 1e-4, False)
            _, _, _, _, both = evaluate_transport(ot_plan_reg, gt_set, C, verbose=False)
            res[sname]['Top 1'].append(both[1]/len(gt_set))
            res[sname]['Top 3'].append(both[3]/len(gt_set))

    sns.set_theme()
    fig, ax = plt.subplots(figsize=(5, 5))
    palette = sns.color_palette(n_colors=len(scenarios) * 2)
    color_idx = 0
    for sname in scenarios:
        markers = ['o', 'X']
        marker_idx = 0
        for metric in ['Top 1', 'Top 3']:
            sns.lineplot(
                x=taus, 
                y=res[sname][metric], 
                marker=markers[marker_idx], 
                label=f'{sname} - {metric}',
                color=palette[color_idx],
                ax=ax,
            )
            marker_idx += 1
        color_idx += 1
    ax.set_xlabel(r'$\tau$')
    ax.set_ylim(0, 1)
    ax.set_title(f"Max Top 1 = {round(np.max([np.max(res[sname]['Top 1']) for sname in res]), 2)}, Top 3 = {round(np.max([np.max(res[sname]['Top 3']) for sname in res]), 2)}")
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_PR_curve(P, gt_set, seuils):
    '''
    Courbe précision-rappel calculée sur le plan de transport, à partir de la probabilité
    d'association obtenue grâce à evaluate_transport_proba.
    Entrées :
        - P : plan de transport.
        - gt_set : vérité de terrain.
        - seuils : liste de seuils s, sur les probabilités d'associations.
    Sortie : 
        - plot de la courbe PR.
    '''
    precision, recall = [], []
    for s in seuils:
        acc, rec = evaluate_transport_proba(P, gt_set, s)
        if acc is not None:
            precision.append(acc)
            recall.append(rec)
    # Plot
    sns.set_theme()
    fig, ax = plt.subplots(figsize=(4,4))
    sns.lineplot(x=recall, y=precision, markers='X', ax=ax)
    ax.set_ylim(0,1)
    ax.set_xlim(0,1)
    ax.set_title('Courbe précision-rappel')
    plt.tight_layout()
    plt.plot()


def compute_costs_barycenter(omim, orpha, node2id, embeddings, deprecated, manifold, weights=None, c=1.):
    w_omim = omim.copy()
    w_orpha = orpha.copy()

    def compute_disease_barycenters(
        profils_omim, node2id, embeddings, deprecated, weights=None, normalize=False, c=1
        ):
        W = torch.from_numpy(embeddings.copy())
        hpo_cols = [c for c in profils_omim.columns if c.startswith('HP')]

        col_meta = {}
        for col in hpo_cols:
            resolved = deprecated.get(col, col)
            if resolved in node2id:
                w = weights[resolved] if (weights is not None and resolved in weights) else 1.0
                col_meta[col] = (W[node2id[resolved]], w)  # (Coordonnée, pondération)

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
            barycenter = fm.frechet_mean(points, c, w)
            barycenters.append(barycenter.numpy())

        profils_omim['barycenter'] = barycenters
        return profils_omim
    w_omim = compute_disease_barycenters(w_omim, node2id, embeddings, deprecated, weights)
    w_orpha = compute_disease_barycenters(w_orpha, node2id, embeddings, deprecated, weights)

    omim_bary = torch.tensor(np.stack(w_omim['barycenter'].values),  dtype=torch.float32)
    orpha_bary = torch.tensor(np.stack(w_orpha['barycenter'].values), dtype=torch.float32)

    n, m = omim_bary.shape[0], orpha_bary.shape[0]
    u = omim_bary.unsqueeze(1).expand(n, m, -1).reshape(n * m, -1)
    v = orpha_bary.unsqueeze(0).expand(n, m, -1).reshape(n * m, -1)

    dists = manifold.sqdist(u, v, c=1)  # (n*m,)
    return dists.reshape(n, m).numpy()


def group_by_length(idx_list, weights_list):
    """Regroupe les indices de disease par taille de support (nb de termes HPO actifs)."""
    groups = defaultdict(list)
    for pos, (idx, w) in enumerate(zip(idx_list, weights_list)):
        if idx:
            groups[len(idx)].append(pos)
    stacked = {}
    for L, positions in groups.items():
        I = np.array([idx_list[p] for p in positions])
        W = np.array([weights_list[p] for p in positions])
        stacked[L] = (np.array(positions), I, W)
    return stacked


# ===========================================================================================
# ================================ Testées et inutilisées ===================================
# ===========================================================================================


def compute_costs_matrix_wasserstein_batched(
    df_omim, df_orpha, 
    node2id_w, 
    embeddings, 
    manifold,
    c,
    weights=None,
    deprecated=data.deprecated,
    S=None,
    gromov=False
    ):
    n = len(df_omim)
    m = len(df_orpha)
    hpo_cols = [c for c in df_omim.columns if c.startswith('HP:')]

    print("Precompute...")
    def precompute(df, weights=weights):
        '''
        Renvoie pour chaque maladie (ligne) du dataframe df la liste des termes HPO actifs et 
        le vecteur de poids uniformes associés.
        '''
        X = df[hpo_cols].to_numpy(dtype=bool)
        resolved_cols = np.array(
            [deprecated.get(col, col) if deprecated.get(col, col) in node2id_w else None for col in hpo_cols], 
            dtype=object)
        valid_mask = resolved_cols != np.array(None)
        X_valid = X[:, valid_mask]
        resolved_valid = resolved_cols[valid_mask]
        terms = [list(resolved_valid[row_mask]) for row_mask in X_valid]
        if weights is None: 
            w = [np.ones(len(t)) / len(t) if t else np.array([]) for t in terms]
        else : 
            w = [np.array([weights[t] for t in term]/np.sum([weights[t] for t in term])) for term in terms]
        return terms, w

    terms_i, weights_i = precompute(df_omim)  # Termes actifs, poids pour les maladies sources
    terms_j, weights_j = precompute(df_orpha)  # Termes actifs, poids pour les maladies destinations
    print("Finished !")

    all_terms = list({h for ts in terms_i + terms_j for h in ts})  # Tous les termes actifs
    term2idx = {h: k for k, h in enumerate(all_terms)}
    hpo_indices = [node2id_w[h] for h in all_terms]  # Indices selon node2id_w
    E = embeddings[hpo_indices]  # Fonctionne si embeddings est construit de la même manière que node2id_w
    if isinstance(E, np.ndarray):
        E = torch.tensor(E, dtype=torch.float32)

    idx_i = [[term2idx[h] for h in ts] for ts in terms_i]  # Index des termes actifs par maladies sources
    idx_j = [[term2idx[h] for h in ts] for ts in terms_j]  # Index des termes actifs par maladies destinations

    C = np.zeros((n, m))

    print("Precomputing full HPO distance matrix...")
    K = E.shape[0]
    D_full = np.zeros((K, K), dtype=np.float32)
    BLOCK = 128  # Réduire si encore OOM (128, 64...)
    for i in tqdm(range(0, K, BLOCK), desc="Distance matrix rows"):
        Ei = E[i:i+BLOCK]          # (b, dim)
        b = Ei.shape[0]
        for j in range(0, K, BLOCK):
            Ej = E[j:j+BLOCK]      # (b2, dim)
            b2 = Ej.shape[0]
                
            Ei_exp = Ei.unsqueeze(1).expand(b, b2, -1).reshape(b * b2, -1)
            Ej_exp = Ej.unsqueeze(0).expand(b, b2, -1).reshape(b * b2, -1)
                
            d = np.sqrt(manifold.sqdist(Ei_exp, Ej_exp, c))
            D_full[i:i+BLOCK, j:j+BLOCK] = d.reshape(b, b2).cpu().numpy()
    
    print(f"HPO distance matrix: {D_full.shape}")
    if S is not None: 
        D_full /= D_full.max()
        simi = S[np.ix_(hpo_indices, hpo_indices)]
        D_full -= D_full * simi  # éventuellement : alpha*simi

    def compute_costs_grouped(
        D_full, idx_i, idx_j, weights_i, weights_j, n, m, 
        epsilon=0.05, device="cuda", max_batch_elems=2_000_000
        ):
        C = np.zeros((n, m), dtype=np.float32)
        groups_i = group_by_length(idx_i, weights_i)   # {L: (positions, I(G,L), W(G,L))}
        groups_j = group_by_length(idx_j, weights_j)
        print(f"{len(groups_i)} tailles distinctes côté OMIM, {len(groups_j)} côté Orpha")

        for Li, (pos_i, I, Wi) in tqdm(groups_i.items(), desc="Length groups (OMIM)"):
            for Lj, (pos_j, J, Wj) in groups_j.items():
                Gi, Gj = I.shape[0], J.shape[0]
                chunk_size = max(1, max_batch_elems // (Li * Lj * Gj))
                for gi_start in range(0, Gi, chunk_size):
                    gi_end = min(gi_start + chunk_size, Gi)
                    I_chunk = I[gi_start:gi_end] 
                    Wi_chunk = Wi[gi_start:gi_end]
                    g = I_chunk.shape[0]
                    cost_block = D_full[I_chunk[:, None, :, None], J[None, :, None, :]]

                    B = g * Gj
                    Ms = torch.tensor(cost_block.reshape(B, Li, Lj), dtype=torch.float32)
                    As = torch.tensor(np.repeat(Wi_chunk, Gj, axis=0), dtype=torch.float32)
                    Bs = torch.tensor(np.tile(Wj, (g, 1)), dtype=torch.float32)

                    costs = compute_transport_sinkhorn_batch(Ms, As, Bs, epsilon, device=device)
                    costs = costs.reshape(g, Gj)

                    rows = pos_i[gi_start:gi_end]
                    C[np.ix_(rows, pos_j)] = costs
        return C
    return compute_costs_grouped(D_full, idx_i, idx_j, weights_i, weights_j, n, m)
