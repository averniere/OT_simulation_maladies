import torch

from poincare import PoincareManifold
from model import Distance_PE

from data import *
from data_utils import *
from OT_utils import * 
from information_content import * 

# OT pipeline
save_dir = "logs/2026_6_18/0/model_final.pt"
manifold = PoincareManifold()
df_orpha_omim_exact = df_orpha_omim[df_orpha_omim["mapping_type"]=='E (Exact mapping: the two concepts are equivalent)']
grid = np.linspace(0.01, 5, 100)

union_diseases = add_corresponding_terms(work_omim, work_orpha, df_orpha_omim)
G_hpo_omim = add_edges(union_diseases, G_hpo_work, depths)
# Cas particulier : on relie manuellement le noeud à son parent dans le sens enfant-parent
G_hpo_omim.add_edge('HP:6001347', 'HP:0001832')

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
        weights = {t:-np.log(w) for t, w in weights.items()}
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
