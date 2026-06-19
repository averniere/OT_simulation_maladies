import torch
import ot
import numpy as np
from scipy.sparse import csgraph
from ot import sinkhorn
from ot.optim import gcg
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed
from poincare import PoincareManifold



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
    """Hamming en pondérant par les embeddings."""
    n = df_omim.shape[0]
    m = df_orpha.shape[0]
    C = np.zeros((n, m))
    print("Compute norms")
    norms, all_hpo = emb_norms(df_omim, df_orpha, node2id_w, model)
    print("Norms computed !")
    
    A = df_omim.reindex(columns=all_hpo, fill_value=0)[all_hpo].values.astype(float)
    B = df_orpha.reindex(columns=all_hpo,  fill_value=0)[all_hpo].values.astype(float)
    #for i_start in tqdm(range(0, n, block_size)):
        #i_end = min(i_start + block_size, n)
        #block = omim_matrix[i_start:i_end]
        #diff = np.abs(block[:, None, :] - orpha_matrix[None, :, :])
        #C[i_start:i_end] = (diff * norms).sum(axis=2)
    Aw = A * norms
    Bw = B * norms

    C = Aw.sum(axis=1)[:, None] + Bw.sum(axis=1)[None, :] - 2 * (A @ Bw.T)
    # Check
    #for i, j in [(0, 0), (3, 7), (9, 14)]:
        #ref = np.dot(np.abs(A[i, :] - B[j, :]), norms)
        #new = C[i, j]
        #print(f"C[{i},{j}]  ref={ref:.6f}  new={new:.6f}  diff={abs(ref-new):.2e}")
    #return C


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


def compute_costs_matrix_wasserstein2(df_omim, df_orpha, node2id_w, model, deprecated, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    n=len(df_omim)
    m=len(df_orpha)
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
        #terms, weights = [], []
        #for _, row in df.iterrows():
            #active = f_active_terms(row, hpo_cols, node2id_w, deprecated)
            #terms.append(active)
            #weights.append(np.ones(len(active)) / len(active) if active else np.array([]))
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
    # E = W[[node2id_w[h] for h in all_terms]]  # Embeddings des termes actifs
    hpo_indices = [node2id_w[h] for h in all_terms]
    E = W[hpo_indices].to(device)

    idx_i = [[term2idx[h] for h in ts] for ts in terms_i]  # Index des termes actifs par maladies sources
    idx_j = [[term2idx[h] for h in ts] for ts in terms_j]  # Index des termes actifs par maladies destinations

    # emb_i = [E[idx] if idx else None for idx in idx_i]  # [Ajout] Vecteurs d'embeddings par maladies sources
    # emb_j = [E[idx] if idx else None for idx in idx_j]  # [Ajout] Vecteurs d'embeddings par maladies destinations

    C = np.zeros((n,m))

    print("Precomputing full HPO distance matrix...")
    # D_full = np.sum((E[:, None, :] - E[None, :, :]) ** 2, axis=-1)  # (K, K)
    K = E.shape[0]
    D_full = np.zeros((K, K), dtype=np.float32)
    BLOCK = 256  # Réduire si encore OOM (128, 64...)
    with torch.no_grad():
        for i in tqdm(range(0, K, BLOCK), desc="Distance matrix rows"):
            Ei = E[i:i+BLOCK]          # (b, dim)
            b = Ei.shape[0]
            
            for j in range(0, K, BLOCK):
                Ej = E[j:j+BLOCK]      # (b2, dim)
                b2 = Ej.shape[0]
                
                Ei_exp = Ei.unsqueeze(1).expand(b, b2, -1).reshape(b * b2, -1)
                Ej_exp = Ej.unsqueeze(0).expand(b, b2, -1).reshape(b * b2, -1)
                
                d = model.manifold.distance(Ei_exp, Ej_exp, model.c)
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


def cost_matrix_hamm(df_omim, df_orpha, weights, block_size=256):
    n = df_omim.shape[0]
    m = df_orpha.shape[0]
    C = np.zeros((n, m))

    hpo_cols = [c for c in df_omim.columns if c.startswith('HP:')]
    all_hpo = list(hpo_cols)
    w = np.array([weights.get(hp, 0.0) for hp in all_hpo])

    A = df_omim.reindex(columns=all_hpo, fill_value=0)[all_hpo].values.astype(float)
    B = df_orpha.reindex(columns=all_hpo,  fill_value=0)[all_hpo].values.astype(float)
    
    # for i_start in tqdm(range(0, n, block_size)):
        # i_end = min(i_start + block_size, n)
        # block = omim_matrix[i_start:i_end]
        # diff = np.abs(block[:, None, :] - orpha_matrix[None, :, :])
        # C[i_start:i_end] = (diff * weights_vector).sum(axis=2)
    Aw = A * w
    Bw = B * w
    C = Aw.sum(axis=1)[:, None] + Bw.sum(axis=1)[None, :] - 2 * (A @ Bw.T)
    return C


def basic_cost_matrix(df_omim, df_orpha, dist_method):
    """
    Inputs:
        - df_omim, df_orpha : dataframes de maladies source et destination.
        - dist_method : 'euclidean', 'hamming', 'jaccard' 
    """
    hpo_cols = [c for c in df_omim.columns if c.startswith('HP:')]
    X = df_omim[hpo_cols].to_numpy().astype(float)
    Y = df_orpha[hpo_cols].to_numpy().astype(float)
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


def Ot_Laplacienne(a, b, xs, xt, M, S, epsilon, eta, numItermax=500, stopThr=1e-9, numInnerItermax=100000,stopInnerThr=1e-9, log=False, verbose=False):
    """
    Inputs :
        - a, b : pondérations de l'information des points sources et destinations à transporter.
        - xs : données sources.
        - xt : données destinations.
        - M : matrice de coûts.
        - S : matrice de similarité.
        - epsilon : régularisation entropique.
        - eta : deuxième régularisation (laplacienne)
    """
    n, m = M.shape
    if a==None:
        a = np.ones(n)/n
    if b==None:
        b = np.ones(m)/m

    #Convertir les entrées en tableaux numpy
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)  # pas forcément utilisé ici
    xt = np.asarray(xt, dtype=np.float64)  # pas forcément utilisé ici
    M = np.asarray(M, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)

    # Calcul du Laplacien (non normé) à partir de la matrice de similarité S
    lS = csgraph.laplacian(S, normed=False)
    lS_sym = 0.5 * (lS + lS.T)  # on le symétrise pour éviter tout problème numérique

    def f(G):
        """
        Calcule la partie "Laplacien" du coût
        sans multiplier par reg2 (le GCG s'en charge).
        """
        # Terme Laplacien
        val_lap = (lS_sym@G)*G
        # si on considere similarité dans la cible egalement avec un param alphe ici = 0.5
        # val_lap = 0.5 * np.trace(G.T.dot(lS2).dot(G)) + 0.5 * np.trace(G.dot(lc2).dot(G.T))
        return val_lap.sum()

    def df(G):
        """
        Gradient de f_lap(G).
        """
        #si on considere similarité dans la cible egalement avec un param alphe ici = 0.5
        #return (ls2 @ G) + (G @ Lc2)
        # Gradient partie laplacienne  2 *  (ls2 @ G)
        return  2 * (lS_sym @ G)

    # Résolution du problème d'optimisation avec l'algorithme du gcg
    return gcg(a, b, M, reg1=epsilon, reg2=eta, f=f, df=df, G0=None, numItermax=numItermax, numItermaxEmd=numInnerItermax, stopThr=stopThr, stopThr2=stopInnerThr,verbose=verbose)
