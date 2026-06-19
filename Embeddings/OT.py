import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import average_precision_score

from poincare import PoincareManifold
from model import Distance_PE

from data import *
from data_utils import *
from OT_utils import * 
from information_content import * 


# Chargement du modèle
save_dir = "logs/2026_6_19/14/model_final.pt"
manifold = PoincareManifold()
df_orpha_omim_exact = df_orpha_omim[df_orpha_omim["mapping_type"]=='E (Exact mapping: the two concepts are equivalent)']
grid = np.linspace(0.01, 5, 100)

# Chargement du graphe complété
union_diseases = add_corresponding_terms(work_omim, work_orpha, df_orpha_omim)
G_hpo_omim = add_edges(union_diseases, G_hpo_work, depths)
# Cas particulier : on relie manuellement le noeud à son parent dans le sens enfant-parent
G_hpo_omim.add_edge('HP:6001347', 'HP:0001832')
objects_omim = list(G_hpo_omim.nodes())
node2id_omim = {n: i for i, n in enumerate(objects_omim)}
edges_omim = np.array([(node2id_omim[u], node2id_omim[v]) for u, v in G_hpo_omim.edges()], dtype=np.int32)

# Métriques de qualité de l'embedding
@torch.no_grad()
def evaluate2(save_dir, edges, node2id, device):
    """
    Retourne MAP et mean rank.
    Corrige : distance avec model.c, calcul GPU, pas de fuite u dans le ranking.
    """
    checkpoint = torch.load(save_dir, map_location='cpu', weights_only=False)
    hp = checkpoint['hyperparams']
    objects = checkpoint['objects']
    model = Distance_PE(
        n=len(objects), dim=hp['dim'], manifold=manifold, sparse=False, 
        learn_curvature=False, init_curvature=1., weight_decay=0
        )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    W = model.weight.to(device)  # (N, dim)

    pos_neighbors = defaultdict(set)
    for u, v in edges:
        pos_neighbors[int(u)].add(int(v))

    ap_scores = []
    ranks_all = []
    N = W.shape[0]
    labels = np.zeros(N)

    for obj in tqdm(objects, desc='Calcul métriques'):
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
    print(map_score, mean_rank)
    return map_score, mean_rank

results = evaluate2(save_dir, edges_omim, node2id_w, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


# OT pipeline
def compare_transport_wasserstein(
    omim, orpha, correspondances, G, node2id, save_dir, manifold, 
    eps1, eps2, baseline='hamming ic', deprecated=deprecated):

    w_omim = omim.reset_index(drop=True)
    w_orpha = orpha.reset_index(drop=True)
    _, valid_omim, valid_orpha = f_ground_truth(w_omim, w_orpha, correspondances)

    w_omim = w_omim[w_omim['database_id'].isin(valid_omim)].reset_index(drop=True)
    w_orpha = w_orpha[w_orpha['database_id'].isin(valid_orpha)].reset_index(drop=True)
    hpo_cols0 = [c for c in w_omim.columns if c.startswith("HP:")]
    gt_set, valid_omim0, valid_orpha0 = f_ground_truth(w_omim, w_orpha, correspondances)
    print(f"Taille OMIM : {w_omim.shape[0]}, ORPHA : {w_orpha.shape[0]}, paires GT : {len(gt_set)}")

    def transport(C, epsilon, gt_set, a=None, b=None):
        print("======== Sans régularisation ========")
        ot_plan, ot_cost = compute_transport(C, a, b)
        ranks, pairs = evaluate_transport(ot_plan, gt_set, C)

        print("======== Avec régularisation ========")
        print(epsilon)
        ot_plan_reg, ot_cots_reg = compute_transport_sinkhorn(C, a, b, epsilon, 10000, 1e-4, False)
        ranks_reg, pairs_reg = evaluate_transport(ot_plan_reg, gt_set, C)
        return ranks_reg, pairs_reg

    if baseline == 'hamming ic':
        # Hamming pondéré par l'information content
        weights, _, _ = compute_information_content(union_diseases, G)
        weights = {t : -np.log(w) for t, w in weights.items()}
        C = cost_matrix_hamm(w_omim, w_orpha, weights)

    base_rank, base_pairs = transport(C, eps1*np.mean(C), gt_set)

    # Wasserstein à deux niveaux
    checkpoint = torch.load(save_dir, map_location='cpu', weights_only=False)
    hp = checkpoint['hyperparams']
    objects = checkpoint['objects']
    model = Distance_PE(
        n=len(objects), dim=hp['dim'], manifold=manifold, sparse=False, 
        learn_curvature=False, init_curvature=1., weight_decay=0
        )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    C_wass = compute_costs_matrix_wasserstein2(w_omim, w_orpha, node2id, model, deprecated)

    wass_rank, wass_pairs = transport(C_wass, eps2*np.mean(C_wass), gt_set)

    return base_rank, base_pairs, wass_rank, wass_pairs

br, bp, wr, wp = compare_transport_wasserstein(
    work_omim, work_orpha, df_orpha_omim, G_hpo_work, node2id_w, 
    save_dir, manifold, grid[2], grid[0])
