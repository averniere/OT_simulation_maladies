import torch
import ot
import numpy as np
from ot import sinkhorn
from tqdm import tqdm
from joblib import Parallel, delayed
from poincare import PoincareManifold
from data_utils import f_active_terms


def compute_cost_matrix(omim, orpha):
    """ 
    Distance de Poincaré entre barycentres.
    """
    omim_bary = torch.tensor(np.stack(omim['barycenter'].values),  dtype=torch.float64)
    orpha_bary = torch.tensor(np.stack(orpha['barycenter'].values), dtype=torch.float64)

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
    W_known = torch.tensor(W[indices], dtype=torch.float64)
    origin = torch.zeros_like(W_known)
    with torch.no_grad():
        hyp_norms = manifold.distance(W_known, origin, c=1.).cpu().numpy()
    norms = np.zeros(len(all_hpo))
    norms[known_pos] = hyp_norms

    return norms, all_hpo


def compute_cost_matrix_pseudo_jacc(df_omim, df_orpha, node2id_w, model, block_size=256):
    n = df_omim.shape[0]
    m = df_orpha.shape[0]
    C = np.zeros((n, m))
    print("Compute norms")
    norms, all_hpo = emb_norms(df_omim, df_orpha, node2id_w, model)
    print("Norms computed !")
    print(all_hpo)

    omim_matrix = df_omim.reindex(columns=all_hpo, fill_value=0)[all_hpo].values.astype(float)
    orpha_matrix = df_orpha.reindex(columns=all_hpo,  fill_value=0)[all_hpo].values.astype(float)
    for i_start in tqdm(range(0, n, block_size)):
        i_end = min(i_start + block_size, n)
        block = omim_matrix[i_start:i_end]
        diff = np.abs(block[:, None, :] - orpha_matrix[None, :, :])
        C[i_start:i_end] = (diff * norms).sum(axis=2)
    # Check
    for i, j in [(0, 0), (3, 7), (9, 14)]:
        ref = np.dot(np.abs(omim_matrix[i, :] - orpha_matrix[j, :]), norms)
        new = C[i, j]
        print(f"C[{i},{j}]  ref={ref:.6f}  new={new:.6f}  diff={abs(ref-new):.2e}")
    return C


def cost_hpos(hpoi, hpoj):
    Ei = torch.tensor(hpoi, dtype=torch.float64).unsqueeze(1)
    Ej = torch.tensor(hpoj, dtype=torch.float64).unsqueeze(0)
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
    E_all_j = np.concatenate(all_emb_j, axis=0)  # (sum_kj, d)
    
    Ei = torch.tensor(emb_i, dtype=torch.float64).unsqueeze(1)       # (ki, 1, d)
    Ej = torch.tensor(E_all_j, dtype=torch.float64).unsqueeze(0)     # (1, sum_kj, d)
    
    manifold = PoincareManifold()
    dists = manifold.distance(Ei, Ej, c=1).detach().numpy()          # (ki, sum_kj)
    
    # Découper selon les tailles
    matrices = []
    start = 0
    for s in sizes_j:
        matrices.append(dists[:, start:start+s])
        start += s
    return matrices


def compute_costs_matrix_wasserstein2(df_omim, df_orpha, node2id_w, model, deprecated):
    n=len(df_omim)
    m=len(df_orpha)
    hpo_cols = [c for c in df_omim.columns if c.startswith('HP:')]
    model.eval()
    W = model.weight.detach().cpu().numpy()
    print("Precompute...")

    def precompute(df):
        '''
        Renvoie pour chaque maladie (ligne) du dataframe df la liste des termes HPO actifs et 
        le vecteur de poids uniformes associés.
        '''
        terms, weights = [], []
        for _, row in df.iterrows():
            active = f_active_terms(row, hpo_cols, node2id_w, deprecated)
            terms.append(active)
            weights.append(np.ones(len(active)) / len(active) if active else np.array([]))
        return terms, weights  

    print("Finished !")
    terms_i, weights_i = precompute(df_omim)  # Termes actifs, poids pour les maladies sources
    terms_j, weights_j = precompute(df_orpha)  # Termes actifs, poids pour les maladies destinations

    all_terms = list({h for ts in terms_i + terms_j for h in ts})  # Tous les termes actifs
    term2idx = {h: k for k, h in enumerate(all_terms)}
    E = W[[node2id_w[h] for h in all_terms]]  # Embeddings des termes actifs

    idx_i = [[term2idx[h] for h in ts] for ts in terms_i]  # Index des termes actifs par maladies sources
    idx_j = [[term2idx[h] for h in ts] for ts in terms_j]  # Index des termes actifs par maladies destinations

    # emb_i = [E[idx] if idx else None for idx in idx_i]  # [Ajout] Vecteurs d'embeddings par maladies sources
    # emb_j = [E[idx] if idx else None for idx in idx_j]  # [Ajout] Vecteurs d'embeddings par maladies destinations

    C = np.zeros((n,m))

    print("Precomputing full HPO distance matrix...")
    D_full = np.sum((E[:, None, :] - E[None, :, :]) ** 2, axis=-1)  # (K, K)
    print(f"HPO distance matrix: {D_full.shape}")

    def compute_row(i):
        if not idx_i[i]:
            return i, np.zeros(len(terms_j))
        # Ei = E[idx_i[i]]
        row = np.zeros(len(terms_j))
        for j in range(len(terms_j)):
            if not idx_j[j]:
                continue
            # Ej = E[idx_j[j]]
            # M  = cost_hpos(Ei, Ej) 
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


def compute_costs_matrix_wasserstein3(df_omim, df_orpha, node2id_w, model, deprecated):
    hpo_cols = [c for c in df_omim.columns if c.startswith('HP:')]
    model.eval()
    W = model.weight.detach().cpu().numpy()

    def precompute(df):
        terms, weights = [], []
        for _, row in df.iterrows():
            active = f_active_terms(row, hpo_cols, node2id_w, deprecated)
            w = np.ones(len(active)) / len(active) if active else np.array([1.0])
            terms.append(active)
            weights.append(w)
        return terms, weights

    terms_i, weights_i = precompute(df_omim)
    terms_j, weights_j = precompute(df_orpha)

    all_terms = list({h for ts in terms_i + terms_j for h in ts})
    term2idx = {h: k for k, h in enumerate(all_terms)}
    E = W[[node2id_w[h] for h in all_terms]]

    idx_i = [[term2idx[h] for h in ts] for ts in terms_i]
    idx_j = [[term2idx[h] for h in ts] for ts in terms_j]

    emb_i = [E[idx] if idx else None for idx in idx_i]
    emb_j = [E[idx] if idx else None for idx in idx_j]

    # Filtrer les j valides une seule fois
    valid_j = [j for j, e in enumerate(emb_j) if e is not None]
    emb_j_valid = [emb_j[j] for j in valid_j]
    weights_j_valid = [weights_j[j] for j in valid_j]

    def compute_row(i):
        row = np.zeros(len(emb_j))
        if emb_i[i] is None:
            return i, row
        
        # Une seule passe torch pour toutes les distances
        # cost_matrices = compute_all_distances(emb_i[i], emb_j_valid)
        
        for k, j in enumerate(valid_j):
            #_, row[j] = compute_transport(
                #cost_matrices[k], weights_i[i], weights_j_valid[k])
            '''    
            projections = [10, 20, 50, 100, 200, 500, 1000]
            n_repeat = 20  # répétitions pour estimer la variance
            results = {}
            for n_proj in projections:
                vals = [ot.sliced_wasserstein_distance(
                    emb_i[i], emb_j[j],
                    a=weights_i[i], b=weights_j[j],
                    n_projections=n_proj, seed=k)
                    for k in range(n_repeat)]
                results[n_proj] = (np.mean(vals), np.std(vals))
                print(f"n={n_proj:>5}  mean={results[n_proj][0]:.4f}  std={results[n_proj][1]:.5f}")    
            '''
            row[j] = ot.sliced_wasserstein_distance(
                emb_i[i], emb_j_valid[k], 
                a=weights_i[i], b=weights_j_valid[k],
                n_projections=100)
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
    optimal_plan = ot.emd(a, b, C)
    optimal_cost = np.sum(optimal_plan*C)
    # print(f"optimal transport plan: \n{optimal_plan}")
    # print(f"Cost matrix: \n{C}")
    # print(f"transport cost: {optimal_cost}")
    return optimal_plan, optimal_cost


def compute_transport_sinkhorn(
    C: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    epsilon: float,
    max_iters: int = 10_000,
    tau: float = 1e-4,
    verbose: bool = False,
    ):
    n = C.shape[0]
    m = C.shape[1]
    if a is None:
        a = np.ones(n)/n
    if b is None:
        b = np.ones(m)/m
    optimal_plan_sinkhorn = sinkhorn(a, b, C, epsilon, numItermax=max_iters, stopThr=tau)
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
                if (i, j_true) not in pairs.keys():
                    pairs[(i, j_true)]=[k, C[i, j_true], P[i, j_true]/marginal[i]]
                    
        if (i, j_true) not in pairs.keys():
            pairs[(i, j_true)]=[0, C[i, j_true], P[i, j_true]/marginal[i]]
            
    n = len(gt_set)
    print(f"Paires évaluées : {n}")
    for k in top_k:
        print(f"Top-{k} accuracy : {results[k]/n:.3f} ({results[k]}/{n})")
    print(f" Rang moyen: {np.mean(ranks):.2f}")

    return ranks, pairs


def plot_consistency(ax, reg_strengths, plan_diff, distance_diff):
    ax[0].loglog(reg_strengths, plan_diff, lw=4)
    ax[0].set_ylabel('$||P^* - P_\epsilon^*||_F$', fontsize=25)
    ax[1].tick_params(which='both', size=20)
    ax[0].grid(ls='--')
    ax[1].loglog(reg_strengths, distance_diff, lw=4)
    ax[1].set_xlabel('Regularization Strength $\epsilon$', fontsize=25)
    ax[1].set_ylabel(r'$ 100 \cdot \frac{\langle C, P^*_\epsilon \rangle - \langle C, P^* \rangle}{\langle C, P^* \rangle} $', fontsize=25)
    ax[1].tick_params(which='both', size=20)
    ax[1].grid(ls='--') 
