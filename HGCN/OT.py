import numpy as np
import torch
import ot

from joblib import Parallel, delayed
from tqdm import tqdm
from data import deprecated


def compute_costs_matrix_wasserstein2(
    df_omim, df_orpha, 
    node2id_w, 
    embeddings, 
    manifold,
    c,
    deprecated=deprecated
    ):
    n = len(df_omim)
    m = len(df_orpha)
    hpo_cols = [c for c in df_omim.columns if c.startswith('HP:')]

    print("Precompute...")
    def precompute(df):
        '''
        Renvoie pour chaque maladie (ligne) du dataframe df la liste des termes HPO actifs et 
        le vecteur de poids uniformes associés.
        '''
        X = df[hpo_cols].to_numpy(dtype=bool)
        resolved_cols = np.array(
            [deprecated.get(col, col) if deprecated.get(col, col) in node2id_w else None for col in hpo_cols], 
            dtype=object)
        valid_mask = resolved_cols != None
        X_valid = X[:, valid_mask]
        resolved_valid = resolved_cols[valid_mask]
        terms = [list(resolved_valid[row_mask]) for row_mask in X_valid]
        weights = [np.ones(len(t)) / len(t) if t else np.array([]) for t in terms]
        return terms, weights

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
    BLOCK = 256  # Réduire si encore OOM (128, 64...)
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

    def compute_row(i):
        if not idx_i[i]:
            return i, np.zeros(len(terms_j))
        # Ei = E[idx_i[i]]
        row = np.zeros(len(terms_j))
        valid_js = [j for j in range(len(terms_j)) if idx_j[j]]
        for j in valid_js:
            # Ej = E[idx_j[j]]
            # M = cost_hpos(Ei, Ej) 
            M = D_full[np.ix_(idx_i[i], idx_j[j])]
            _, row[j] = compute_transport(M, weights_i[i], weights_j[j])
        return i, row
    
    results = Parallel(n_jobs=-1)(
        delayed(compute_row)(i) for i in tqdm(range(len(df_omim)), desc="OMIM")
    )
    C = np.zeros((len(df_omim), len(df_orpha)))
    for i, row in results:
        C[i] = row
    return C

    
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


def evaluate_transport(P, gt_set, C, top_k=(1, 3, 5)):
    """
    Évalue le plan de transport P contre la vérité terrain.
    Inputs : 
        - P : plan de transport ;
        - gt_set : correspondances exactes entre les maladies des deux bases de données sous la forme
        {(i_1,j_1), (i_2, j_2)...} ;
        - C : matrice de coût ;
        - top_k : précision, j_true est au plus la k-ième destination recevant le plus de masse.
    """
    results = {k: 0 for k in top_k}
    pairs = {}
    ranks = []
    marginal = np.sum(P, axis=1)

    for (i, j_true) in gt_set:
        # Colonnes triées par masse décroissante pour la ligne i
        ranked_cols = np.argsort(P[i])[::-1]
        rank = np.where(ranked_cols == j_true)[0]
        if len(rank) == 0:
            continue
        rank = rank[0] + 1
        ranks.append(rank) # Rang de la vraie maladie j_true dans la matrice de transport
        
        for k in top_k:
            if rank <= k:
                results[k] += 1
                if C is not None and (i, j_true) not in pairs.keys():
                    pairs[(i, j_true)]=[k, C[i, j_true], P[i, j_true]/marginal[i]]
                    
        if C is not None and (i, j_true) not in pairs.keys():
            pairs[(i, j_true)]=[0, C[i, j_true], P[i, j_true]/marginal[i]]
            
    n = len(gt_set)
    print(f"Paires évaluées : {n}")
    for k in top_k:
        print(f"Top-{k} accuracy : {results[k]/n:.3f} ({results[k]}/{n})")
    print(f" Rang moyen: {np.mean(ranks):.2f}")

    return ranks, pairs


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

