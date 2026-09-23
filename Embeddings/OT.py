import torch
import ot
import numpy as np
import pandas as pd
from scipy.sparse import csgraph
import seaborn as sns
import matplotlib.pyplot as plt

from numpy.linalg import norm
from scipy import sparse
from ot import sinkhorn
from ot.optim import gcg
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed
from collections import defaultdict
from poincare import PoincareManifold
from data_utils import f_ground_truth, f_ground_truth_broad


#===========================================================================================
#================================== Matrices de coût =======================================
#===========================================================================================


def compute_cost_matrix(omim, orpha, colname='barycenter'):
    """ 
    Distance de Poincaré entre barycentres.
    """
    omim_bary = torch.tensor(np.stack(omim[colname].values),  dtype=torch.float32)
    orpha_bary = torch.tensor(np.stack(orpha[colname].values), dtype=torch.float32)

    n, m = omim_bary.shape[0], orpha_bary.shape[0]

    u = omim_bary.unsqueeze(1).expand(n, m, -1).reshape(n * m, -1)
    v = orpha_bary.unsqueeze(0).expand(n, m, -1).reshape(n * m, -1)

    manifold = PoincareManifold()
    dists = manifold.distance(u, v, c=1)  # (n*m,)

    return dists.reshape(n, m).numpy()


def emb_norms(df_omim, df_orpha, node2id_w, model, manifold=PoincareManifold()):
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


def cost_hpos(hpoi, hpoj):
    Ei = torch.tensor(hpoi, dtype=torch.float32).unsqueeze(1)
    Ej = torch.tensor(hpoj, dtype=torch.float32).unsqueeze(0)
    manifold = PoincareManifold()
    dists = manifold.distance(Ei, Ej, c=1)
    return dists.detach().numpy()


def compute_all_distances(emb_i, all_emb_j):
    """
    emb_i : np.array (ki, d)
    all_emb_j : liste de np.array (kj, d)
    Retourne une liste de matrices de distances
    """
    # Concaténer tous les embeddings j
    sizes_j = [len(e) for e in all_emb_j]
    E_all_j = np.concatenate(all_emb_j, axis=0)
    
    Ei = torch.tensor(emb_i, dtype=torch.float32).unsqueeze(1)
    Ej = torch.tensor(E_all_j, dtype=torch.float32).unsqueeze(0)
    
    manifold = PoincareManifold()
    dists = manifold.distance(Ei, Ej, c=1).detach().numpy()
    
    # Découper selon les tailles
    matrices = []
    start = 0
    for s in sizes_j:
        matrices.append(dists[:, start:start+s])
        start += s
    return matrices


def compute_costs_matrix_wasserstein2(
    df_omim, 
    df_orpha, 
    node2id_w, 
    model, 
    deprecated, 
    unbalanced=False,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
    n = len(df_omim)
    m = len(df_orpha)
    hpo_cols = [c for c in df_omim.columns if c.startswith('HP:')]
    model.eval()
    # W = model.weight.detach().cpu().numpy()
    W = model.weight.detach()

    print("Precompute...")
    def precompute(df):
        '''
        Renvoie pour chaque maladie (ligne) du dataframe df la liste des termes HPO actifs et 
        le vecteur de poids uniformes associés.
        '''
        X = df[hpo_cols].to_numpy(dtype=np.float32)
        resolved_cols = np.array([deprecated.get(col, col) if deprecated.get(col, col) in node2id_w else None for col in hpo_cols], dtype=object)
        valid_mask = resolved_cols != None
        X_valid = X[:, valid_mask]
        resolved_valid = resolved_cols[valid_mask]

        terms = []
        weights = []
        for row in X_valid:
            active_mask = row > 0 
            row_terms = list(resolved_valid[active_mask])
            row_values = row[active_mask]
            total = row_values.sum()
            row_weights = row_values / total if total > 0 else np.array([])
            terms.append(row_terms)
            weights.append(row_weights)
        return terms, weights

    terms_i, weights_i = precompute(df_omim)  # Termes actifs, poids pour les maladies sources
    terms_j, weights_j = precompute(df_orpha)  # Termes actifs, poids pour les maladies destinations
    print("Finished !")

    all_terms = list({h for ts in terms_i + terms_j for h in ts})  # Tous les termes actifs
    term2idx = {h: k for k, h in enumerate(all_terms)}
    # E = W[[node2id_w[h] for h in all_terms]]  # Embeddings des termes actifs
    hpo_indices = [node2id_w[h] for h in all_terms]
    E = W[hpo_indices].to(device)

    idx_i = [[term2idx[h] for h in ts] for ts in terms_i]  # Index des termes actifs par maladies sources
    idx_j = [[term2idx[h] for h in ts] for ts in terms_j]  # Index des termes actifs par maladies destinations

    C = np.zeros((n, m))

    print("Precomputing full HPO distance matrix...")
    K = E.shape[0]
    D_full = np.zeros((K, K), dtype=np.float32)
    BLOCK = 256  # On calcule par blocs pour éviter d'exploser la mémoire
    with torch.no_grad():
        for i in tqdm(range(0, K, BLOCK), desc="Distance matrix rows"):
            Ei = E[i:i+BLOCK]
            b = Ei.shape[0]
            
            for j in range(0, K, BLOCK):
                Ej = E[j:j+BLOCK]
                b2 = Ej.shape[0]
                
                Ei_exp = Ei.unsqueeze(1).expand(b, b2, -1).reshape(b * b2, -1)
                Ej_exp = Ej.unsqueeze(0).expand(b, b2, -1).reshape(b * b2, -1)
                
                d = model.manifold.distance(Ei_exp, Ej_exp, model.c)
                D_full[i:i+BLOCK, j:j+BLOCK] = d.reshape(b, b2).cpu().numpy()
    
    print(f"HPO distance matrix: {D_full.shape}")
    
    def self_transport(idx, w):
        M = D_full[np.ix_(idx, idx)]
        reg = 0.1 * np.mean(M)
        _, val = compute_transport_sinkhorn(M, w, w, reg)
        return val

    #b_cache = {i: self_transport(idx_i[i], weights_i[i]) for i in valid_is}
    #c_cache = {j: self_transport(idx_j[j], weights_j[j]) for j in valid_js}

    def compute_row(i):
        if not idx_i[i]:
            return i, np.zeros(len(terms_j))
        row = np.zeros(len(terms_j))
        valid_js = [j for j in range(len(terms_j)) if idx_j[j]]
        for j in valid_js:
            M_ij = D_full[np.ix_(idx_i[i], idx_j[j])]
            #M_mean = np.mean(M_ij)
            #reg = 0.1*M_mean
            if unbalanced:
                eps = 0.1*np.mean(M_ij)
                _, row[j] = compute_unbalanced(M_ij, None, None, eps, 0.9, 0.9)
            else:
                _, row[j] = compute_transport(M_ij, weights_i[i], weights_j[j])
            #_, a = compute_transport_sinkhorn(M_ij, weights_i[i], weights_j[j], reg)
            #row[j] = a - (b_cache[i]+c_cache[j])/2
        return i, row
    
    results = Parallel(n_jobs=-1)(
        delayed(compute_row)(i) for i in tqdm(range(len(df_omim)), desc="OMIM")
    )
    C = np.zeros((len(df_omim), len(df_orpha)))
    for i, row in results:
        C[i] = row
    return C


def cost_matrix_hamm(df_omim, df_orpha, weights, block_size=256):
    '''
    Entrées :
        - df_omim, df_orpha : deux bases de maladies annotées,
        - weights : dictionnaire de la forme {terme HPO: pondération}
    Sortie :
        - Matrice des dissimilarités de Hamming pondérées entre paires de maladies.
    '''
    n = df_omim.shape[0]
    m = df_orpha.shape[0]
    C = np.zeros((n, m))

    hpo_cols = [c for c in df_omim.columns if c.startswith('HP:')]
    all_hpo = list(hpo_cols)
    w = np.array([weights.get(hp, 0.0) for hp in all_hpo])

    A = df_omim.reindex(columns=all_hpo, fill_value=0)[all_hpo].values.astype(np.float32)
    B = df_orpha.reindex(columns=all_hpo,  fill_value=0)[all_hpo].values.astype(np.float32)
    
    A = (A > 0).astype(np.float32)  # Transformation en des datasets binaires dans le cas où on a des fréquences
    B = (B > 0).astype(np.float32)  # Transformation en des datasets binaires dans le cas où on a des fréquences
    
    A_sp  = sparse.csr_matrix(A)
    Bw_sp = sparse.csr_matrix(B * w)

    inter = A_sp.dot(Bw_sp.T)
    inter = inter.toarray()

    Aw_sum = A_sp.multiply(w).sum(axis=1).A1 
    Bw_sum = Bw_sp.sum(axis=1).A1

    C = Aw_sum[:, None] + Bw_sum[None, :] - 2 * inter
    #Aw = A * w
    #Bw = B * w
    #inter = A @ Bw.T
    #C = Aw.sum(axis=1)[:, None] + Bw.sum(axis=1)[None, :] - 2 * (inter)
    return C


def basic_cost_matrix(df_omim, df_orpha, dist_method):
    """
    Entrées:
        - df_omim, df_orpha : dataframes de maladies source et destination.
        - dist_method : 'euclidean', 'hamming', 'jaccard' 
    """
    hpo_cols = [c for c in df_omim.columns if c.startswith('HP:')]
    X = df_omim[hpo_cols].to_numpy().astype(np.float32)
    Y = df_orpha[hpo_cols].to_numpy().astype(np.float32)
    if dist_method == 'euclidean':
        distance_matrix = pairwise_distances(X, Y, metric=dist_method, n_jobs=-1)
    if dist_method == 'jaccard':
        inter = X @ Y.T
        union = X.sum(axis=1)[:, None]+Y.sum(axis=1)[None, :]-inter
        distance_matrix = 1 - np.where(union == 0, 1.0, inter/union)
    if dist_method == 'hamming':
        inter = X @ Y.T
        distance_matrix = X.sum(axis=1)[:, None]+Y.sum(axis=1)[None, :] - 2*inter
    print("Check :", distance_matrix.shape)
    return distance_matrix


def pearson_corr(A, B, eps=1e-10):
    A_mean = A - np.mean(A, axis=1, keepdims=True)
    B_mean = B - np.mean(B, axis=1, keepdims=True)
    A_norm = A_mean / np.maximum(np.linalg.norm(A_mean, axis=1, keepdims=True), eps)
    B_norm = B_mean / np.maximum(np.linalg.norm(B_mean, axis=1, keepdims=True), eps)

    return A_norm @ B_norm.T


def correlation_cost(df1, df2):
    hpo_cols = [c for c in df1.columns if c.startswith('HP')]
    X = df1[hpo_cols].to_numpy(dtype=float)
    Y = df2[hpo_cols].to_numpy(dtype=float)
    C = pearson_corr(X, Y)
    return 1-C


def resnik_disease_similarities(df_omim, df_orpha, resnik_hpos, method):
    n = df_omim.shape[0]
    m = df_orpha.shape[0]

    colnames = [c for c in df_omim.columns if c.startswith('HP:')]
    X = df_omim[colnames].values
    Y = df_orpha[colnames].values
    n_active_x = X.sum(axis=1)
    n_active_y = Y.sum(axis=1)

    M = np.exp(-resnik_hpos)  # Matrice de similarités entre termes
    T = X.shape[1]
    d_max = M.max()
    
    if method == 'sum':
        C = X @ M @ Y.T

    if method == 'mean':
        S = X @ M @ Y.T
        n_active_x = X.sum(axis=1, keepdims=True)
        n_active_y = Y.sum(axis=1, keepdims=True)
        denom = n_active_x @ n_active_y.T
        denom[denom == 0] = 1
        return S / denom

    Q = np.full((n, T), d_max*2)
    for i in tqdm(range(n)):
        active_i = np.where(X[i] == 1)[0]  # termes actifs de la maladie i
        if len(active_i) == 0:
            continue
        Q[i, :] = M[active_i, :].min(axis=0)
    
    R = np.full((T, m), d_max*2)
    for j in tqdm(range(m)):
        active_j = np.where(Y[j] == 1)[0]  # termes actifs de la maladie j
        if len(active_j) == 0:
                continue
        R[:, j] = M[:, active_j].min(axis=1)

    row_sum = X @ np.where(np.isinf(R), d_max*2, R)
    row_best_mean = row_sum / np.maximum(n_active_x[:, None], 1)

    col_sum = np.where(np.isinf(Q), d_max*2, Q) @ Y.T
    col_best_mean = col_sum / np.maximum(n_active_y[None, :], 1)

    if method == 'funSimAvg':
        return (row_best_mean + col_best_mean) / 2

    if method == 'funSimMax':
        return np.minimum(row_best_mean, col_best_mean)  # distance : min = meilleur

    if method == 'bma':
        denom = n_active_x[:, None] + n_active_y[None, :]
        return (row_sum + col_sum) / denom

    if method == 'max':
        C = np.full((n, m), np.inf)
        for i in range(n):
            active_i = np.where(X[i] == 1)[0]
            if len(active_i) == 0:
                continue
            C[i, :] = R[active_i, :].min(axis=0)
        C = np.where(np.isinf(C), d_max*2, C)
        return C

    raise ValueError(f"Méthode inconnue : {method}")

#===========================================================================================
#=============================== Méthodes de transport =====================================
#===========================================================================================


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
    log: bool = False
    ):
    n = C.shape[0]
    m = C.shape[1]
    if a is None:
        a = np.ones(n)/n
    if b is None:
        b = np.ones(m)/m
    assert np.isclose(a.sum(), 1.0), f"somme a = {a.sum()}"
    assert np.isclose(b.sum(), 1.0), f"somme b = {b.sum()}"
    if log:
        optimal_plan_sinkhorn = ot.bregman.sinkhorn_stabilized(a, b, C, epsilon, numItermax=max_iters, stopThr=tau)
    else:
        optimal_plan_sinkhorn = sinkhorn(a, b, C, epsilon, numItermax=max_iters, stopThr=tau)
        #optimal_plan_sinkhorn = ot.bregman.sinkhorn_epsilon_scaling(a, b, C, )
    optimal_cost_sinkhorn = np.sum(optimal_plan_sinkhorn*C)

    if verbose:
        print(f"entropic optimal transport plan: \n{optimal_plan_sinkhorn}")
        print(f"entropic transport cost: {optimal_cost_sinkhorn}")

    return optimal_plan_sinkhorn, optimal_cost_sinkhorn


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


#===========================================================================================
#============================== Evaluation des résultats ===================================
#===========================================================================================
def read_transport_plan(P, gt_set):
    ranks = {}
    for (i,j) in gt_set:
        ranked_cols = np.argsort(P[i])[::-1]
        ranked_lines = np.argsort(P[:,j])[::-1]
            
        rankj = np.where(ranked_cols == j)[0]
        if len(rankj) == 0:
            continue
        rankj = rankj[0] + 1
        
        ranki = np.where(ranked_lines == i)[0]
        if len(ranki) == 0:
            continue
        ranki = ranki[0] + 1
        ranks[(i, j)] = [rankj, ranki]
    return ranks


def evaluate_transport(P, gt_set, C, top_k=(1, 3, 5), verbose=True):
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

    return ranks_orpha, ranks_omim, pairs1, pairs2, both


def evaluate_transport_broad(P, df1, df2, correspondances, verbose=True, tol=1.5):
    '''
    Entrées :
        - P : plan de transport.
        - df1, df2 : les deux bases d'annotations entre lesquelles on résout le problème.
        - correspondances : base de correspondances entre les maladies.
        - tol : tolérance sur le rang de la maladie dans la matrice de transport. Si une maladie
        M correspond à 2 maladies dans la base opposée, alors on regarde les tol*2 = 3 maladies
        recevant le plus de masse dans le plan de transport.
    Sortie : 
        - both : dictionnaire indiquant pour chaque paire si les maladies sont correctement
        retrouvées (True) en lisant le plan de transport en lignes et en colonnes, pour un niveau
        de tolérance 0 et tol.
        - ranks_js, ranks_is : rangs des maladies Orphanet et OMIM respectivement dans le plan 
        de transport.
    '''
    gt_set1, gt_set2 = f_ground_truth_broad(df1, df2, correspondances)
    gt_set, _, _ = f_ground_truth(df1, df2, correspondances)
    both = {t : {(i,j):[False, False] for (i,j) in gt_set} for t in (0, tol)}
    score_i = 0
    score_j = 0 
    err_rankj, err_ranki = [], []
    for i, js in gt_set1.items():
        ranks_js = []
        ranked_cols = np.argsort(P[i])[::-1]
        m = len(js)
        m_tol = tol*m
        for j in js: 
            rank_j = np.where(ranked_cols == j)[0]
            if len(rank_j) == 0:
                continue
            rank_j = rank_j[0] + 1
            if rank_j <= m: 
                both[0][(i,j)][0] = True
                both[tol][(i,j)][0] = True
                score_i += 1
            elif rank_j > m and rank_j<= m_tol:
                both[tol][(i,j)][0] = True
            ranks_js.append(rank_j)
        err_rankj.append(np.sum(ranks_js)-(m*(m+1))/2)
    for j, i_s in gt_set2.items():
        ranks_is = []
        ranked_lines = np.argsort(P[:,j])[::-1]
        m = len(i_s)
        m_tol = tol*m
        for i in i_s:
            rank_i = np.where(ranked_lines == i)[0]
            if len(rank_i) == 0:
                continue
            rank_i = rank_i[0] + 1
            if rank_i <= m:
                both[0][(i,j)][1] = True
                both[tol][(i,j)][1] = True
                score_j += 1
            elif rank_i>m and rank_i<=m_tol:
                both[tol][(i,j)][1] = True
            ranks_is.append(rank_i)
        err_ranki.append(np.sum(ranks_is)-(m*(m+1))/2)

    if verbose:
        print(f"Paires évaluées : {len(gt_set)}")
        for val in ["Lignes", "Colonnes", "Lignes et Colonnes"]:
            print(f"=========== {val} ===========")
            if val == "Lignes":
                print(f"Paires correctes : {score_i/len(gt_set):.3f} ({score_i}/{len(gt_set)})")
                print(f"Erreur moyenne sur le rang : {np.mean(err_rankj):.0f}/{len(df2)} (absolue : {np.sum(err_ranki)})")
            elif val == "Colonnes":
                print(f"Paires correctes : {score_j/len(gt_set):.3f} ({score_j}/{len(gt_set)})")
                print(f"Erreur moyenne sur le rang : {np.mean(err_ranki):.0f}/{len(df1)} (absolue : {np.sum(err_rankj)})")
            else:
                print(f"Paires correctes : {np.sum([both[0][(i,j)] == [True, True] for (i,j) in gt_set])/len(gt_set):.3f} ({np.sum([both[0][(i,j)] == [True, True] for (i,j) in gt_set])}/{len(gt_set)})")
                print(f"Paires correctes avec une tolérance de {tol} : {np.sum([both[tol][(i,j)] == [True, True] for (i,j) in gt_set])/len(gt_set):.3f} ({np.sum([both[tol][(i,j)] == [True, True] for (i,j) in gt_set])}/{len(gt_set)})")
    return both, ranks_is, ranks_js


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


def plot_unbalanced(taus, C, epsilon, gt_set, ax=None, title=None, legend=True):
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
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5,5))
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
    ax.set_title(f" {title or ""}- Max Top 1 = {round(np.max([np.max(res[sname]['Top 1']) for sname in res]), 2)}, Top 3 = {round(np.max([np.max(res[sname]['Top 3']) for sname in res]), 2)}")
    if not legend:
        ax.legend().remove()

    if standalone:
        plt.tight_layout()
        plt.show()

    return ax


def plot_PR_curve(P, gt_set, seuils, ax=None, title=None):
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
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(4,4))

    sns.lineplot(x=recall, y=precision, marker='X', ax=ax)
    ax.set_ylim(0,1)
    ax.set_xlim(0,1)
    ax.set_title(title or 'Courbe précision-rappel')

    if standalone:
        plt.tight_layout()
        plt.show()

    return ax

#===========================================================================================
#============================== Régularisation laplacienne =================================
#===========================================================================================


def simi_ppi(df, m_to_idx):
    '''
    Entrées : 
        - Dataframe récapitulant les données sur les maladies, associations de gènes et PPI.
        - m_to_idx : dictionnaire {maladie: index} du dataframe à partir duquel on calculera
        le plan de transport (garde-fou).
    '''
    n = df.shape[0]
    protein_to_idx = defaultdict(list)
    for i, row in df.iterrows():
        proteins = row['protein'] if isinstance(row['protein'], list) else [row['protein']]
        for p in proteins:
            if pd.notna(p):
                protein_to_idx[p].append(row['disease_id'])

    S = np.zeros((n, n))
    for i, row in df.iterrows():
        m_i = row['database_id']
        ind_i = m_to_idx[m_i]
        interactions = list(row['protein2'])
        scores = row['combined_score'] if isinstance(row['combined_score'], list) else []
        for p, val in zip(interactions, scores):
            if pd.isna(p) or pd.isna(val):
                continue
            for m_j in protein_to_idx.get(p, []):
                ind_j = m_to_idx[m_j]
                S[ind_i, ind_j] = max(val/1000, 0)
                S[ind_j, ind_i] = max(val/1000, 0)
    np.fill_diagonal(S, 1.)
    return S


def laplacian(x):
    r"""Compute Laplacian matrix"""
    L = np.diag(np.sum(x, axis=1)) - x
    return L


def otda(
    a, b, 
    df_source, df_target, 
    C, 
    Ss, St,
    epsilon,
    eta=1., 
    alpha=0.5, 
    numItermax=25,
    stopThr=1e-9,
    numInnerItermax=100_000,
    stopInnerThr=5e-8,
    verbose=True
    ):
    '''
    Reprend et adapte le code de la fonction emd_laplace de la dépendance ot.da du package POT.
    cf : https://github.com/PythonOT/POT/blob/master/ot/da.py 
    '''

    n, m = C.shape
    if a is None:
        a = np.ones(n)/n
    if b is None:
        b = np.ones(m)/m

    hpo_cols = [c for c in df_source.columns if c.startswith('HP')]
    xs = np.asarray(df_source[hpo_cols], dtype=np.float32)
    xt = np.asarray(df_target[hpo_cols], dtype=np.float32)

    lS = laplacian(Ss)
    lT = laplacian(St)

    ls2 = lS + lS.T
    lt2 = lT + lT.T
    xt2 = np.dot(xt, xt.T)
    xs2 = np.dot(xs, xs.T)

    def f(G):
        #return alpha * np.trace(G.T @ lS @ G) + (1 - alpha) * np.trace(G @ lT @ G.T)
        return alpha * np.trace(xt.T @ G.T @ lS @ G @ xt) + (1 - alpha) * np.trace(xs.T @ G @ lT @ G.T @ xs)

    def df(G):
        #return alpha * (ls2 @ G) + (1-alpha) * (G @ lt2)
        return alpha * (ls2 @ G @ xt2) + (1 - alpha) * (xs2 @ G @ lt2)

    assert (C >= 0).all(), "C contient des valeurs négatives !"
    G_emd = ot.sinkhorn(a, b, C, reg=epsilon*np.mean(C), numItermax=10_000)
    print("G0 nan:", np.isnan(G_emd).any())
    G0Xt = G_emd @ xt
    term1 = np.einsum('ij,ij->', lS @ G0Xt, G0Xt)
    G0tXs = G_emd.T @ xs
    term2 = np.einsum('ij,ij->', lT @ G0tXs, G0tXs)
    f_Gemd = alpha * term1 + (1 - alpha) * term2

    print("Omega_c:", f_Gemd) 
    print("Sinkhorn :", np.sum(G_emd * C))
    print("Coefficient multiplicateur :", np.sum(G_emd * C) / f_Gemd)
    eta_calibrated = np.sum(G_emd*C)/f_Gemd
    return ot.optim.gcg(
        a,
        b,
        C,
        reg1=eta,
        reg2=epsilon,
        f=f,
        df=df,
        G0=None,
        numItermax=numItermax, 
        numInnerItermax=numInnerItermax, 
        stopThr=stopThr, 
        stopThr2=stopInnerThr, 
        verbose=verbose
    )


def plot_laplace(df1, df2, etas, C, regs, Ss, St, gt_set, alpha, numItermax=25):
    '''
    Entrées :
        - df1, df2 : les deux bases d'annotations entre lesquelles on calcule le transport.
        - etas : liste des eta à tester.
        - C : matrice de coût.
        - regs : liste des epsilon à tester.
        - Ss, St : matrices de similarité source et destination.
        - gt_set : vérité de terrain.
        - alpha : paramètre alpha dans la régularisation laplacienne.
        - numItermax : nombre d'itérations maximales du gcg.
    Représente l'évolution des résultats en fonction de eta.
    '''
    res = {eps:{'Top 1':[], 'Top 3':[]} for eps in regs}
    for eps in tqdm(res.keys()):
        epsilon0 = eps * np.mean(C)
        ot_plan_reg, _ = compute_transport_sinkhorn(
            C, None, None, epsilon0, 10000, 1e-4, False
            )
        _, _, _, _, both = evaluate_transport(ot_plan_reg, gt_set, C)
        res[eps]['Top 1'].append(both[1]/len(gt_set))
        res[eps]['Top 3'].append(both[3]/len(gt_set))
        for eta in etas:
            ot_lapl = otda(
                None, None,  # a, b
                df1, df2, 
                C, 
                Ss, St,
                epsilon0, 
                eta, 
                alpha,
                numItermax=numItermax
                )

            print("Transport computed !")
            _, _, _, _, both = evaluate_transport(ot_lapl, gt_set, C, verbose=False)
            res[eps]['Top 1'].append(both[1]/len(gt_set))
            res[eps]['Top 3'].append(both[3]/len(gt_set))
    sns.set_theme()
    fig, ax = plt.subplots(figsize=(5, 5))
    palette = sns.color_palette(n_colors=len(regs) * 2)
    color_idx = 0
    for eps in regs:
        markers = ['o', 'X']
        marker_idx = 0
        for metric in ['Top 1', 'Top 3']:
            ax.axhline(
                y=res[eps][metric][0], 
                color=palette[color_idx], 
                linestyle='--', 
                linewidth=1, 
                alpha=0.7, 
                label=f'{eps} - {metric} (Init)'
                )
            sns.lineplot(
                x=etas, 
                y=res[eps][metric][1:], 
                marker=markers[marker_idx], 
                label=f'{eps} - {metric}',
                color=palette[color_idx],
                ax=ax,
                )
            marker_idx += 1
        color_idx += 1
    ax.set_xlabel(r'$\eta$')
    ax.set_xscale('log')
    ax.set_ylim(0, 1)
    ax.set_title(f"Max Top 1 = {round(np.max([np.max(res[eps]['Top 1']) for eps in res]), 2)}, Top 3 = {round(np.max([np.max(res[eps]['Top 3']) for eps in res]), 2)}")
    ax.legend()

    plt.tight_layout()
    plt.show()