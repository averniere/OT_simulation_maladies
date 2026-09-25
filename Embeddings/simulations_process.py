import os
import torch
import numpy as np
import pandas as pd
import sys
import data 
import model
import poincare
import simulations as simi

from information_content import compute_information_content
from simulations_add_metrics import add_recall_precision

# Chargement des données
weights, diseases, all_diseases = compute_information_content(data.profils_omim, data.G_hpo_work)
ic = {t: -np.log(weights[t]) if weights.get(t, 0) > 0 else 0.0 for t in weights}

checkpoint = torch.load('logs/2026_5_7/12/model_final.pt', map_location='cpu', weights_only=False)
objects = checkpoint['objects']
hp = checkpoint['hyperparams']
model = model.Distance_PE(
    n=len(data.objects_w), dim=hp['dim'],
    manifold=poincare.PoincareManifold(), sparse=False, 
    learn_curvature=False, init_curvature=1., 
    weight_decay=0
    )
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

n_replicats = 3  # Nombre de simulations
result_dir = "simuls"
python_path = sys.executable

# Paramètres
quantile_list = [0., 0.5] + list(np.arange(0.75, 0.95, 0.03))
n_complex_list = [30]  # Nombre de maladies complexes à simuler           
n_match_list = [50]  # [50, 100, 150] Nombre de maladies mendéliennes par maladie complexe              
noise_levels = [0., 0.2, 0.5]  # [0., 0.2]   
epsilon = 0.05
overlap_test = [0.2, 0.5]
group_size = 6
eta_list = [1e3] # , 1e4]
cost_method = 'jaccard'  # 'wasserstein', 'hamming pondéré', 'hamming pondéré normes', 'hamming', 'jaccard', 'pearson correlation'
weights_cost = ic
transp_method_list = ['classic', 'unbalanced']

# S'assurer que le dossier existe
os.makedirs(result_dir, exist_ok=True)

if os.path.exists("simuls/results_to_r.csv"):
    os.remove("simuls/results_to_r.csv")
if os.path.exists("simuls/simu_brut.csv.gz"):
    os.remove("simuls/simu_brut.csv.gz")

for k in range(n_replicats):
    if os.path.exists(f"simuls/results_{k+1}.csv"):
        os.remove(f"simuls/results_{k+1}.csv")

# Étape 1 et 2 : Exécuter les simulations et stocker les résultats
for i in range(n_replicats):
    print(f"Lancement de la simulation {i+1}/{n_replicats}")

    # Lancer la simulation
    #subprocess.run([python_path, "simulations.py"], check=True)
    results, df_truth, df_target = simi.process_simulation(
        source_data=data.profils_omim[data.hpo_cols0],
        n_complex_list=n_complex_list,  # Nombre de maladies complexes à simuler           
        n_match_list=n_match_list,  # [10 , 50], Nombre de maladies mendéliennes par maladie complexe              
        noise_levels=noise_levels,  # [0, 0.05, 0.1, 0.2]   
        quantiles=quantile_list,
        epsilon=epsilon,
        overlap_test=overlap_test,    
        group_size=group_size,                     
        eta_list=eta_list,
        model=model,
        node2id=data.node2id_w,
        deprecated=data.deprecated,
        cost_method=cost_method,  # 'wasserstein', 'hamming pondéré', 'hamming pondéré normes', 'hamming', 'jaccard', 'pearson correlation'
        weights_cost=ic,  # Pondération si 'hamming pondéré'
        transp_method_list=transp_method_list  # 'classic' ou 'unbalanced'
        )

    #results.to_csv('simuls/simu_brut.csv.gz', sep=';', index=False, compression="gzip")
    df_target.to_csv("simuls/target.csv", sep=';', index=False)
    df_truth.to_csv("simuls/truth.csv", sep=';', index=False)
    
    
    # Appliquer add_metrics
    # subprocess.run([python_path, "simulations_add_metrics.py"], check=True)
    #simu = pd.read_csv('simuls/simu_brut.csv.gz', sep=';')
    add_recall_precision(results, quantile_list)
    columns_to_drop = [f'Associations_quantile_{q}' for q in quantile_list]
    recall_prec = results.drop(columns=['Complex_Disease', 'Mendelian_Sources'] + columns_to_drop)
    # Colonnes sur lesquelles regrouper
    group_cols = ['n_match', 'OT_type', 'noise_level', 'overlap_rate', 'n_complex', 'eta']
    # Calcul des moyennes
    result = recall_prec.groupby(group_cols, as_index=False).mean()
    result.to_csv('simuls/simul.csv')
    
    # Renommer et stocker le fichier resultats.csv
    result_file = os.path.join(result_dir, f"results_{i+1}.csv")
    os.rename("simuls/simul.csv",result_file)

    # renommer et stocker les fichiers target
    target_file = os.path.join(result_dir, f"target_{i+1}.csv")
    os.rename("simuls/target.csv", target_file)

    # renommer et stocker les fichiers truth
    truth_file = os.path.join(result_dir, f"truth_{i+1}.csv")
    os.rename("simuls/truth.csv", truth_file)
    
    # Supprimer le fichier volumineux de base
    #os.remove("simuls/simu_brut.csv.gz")
    #os.remove("simuls/target.csv")


# Étape 3 : Calcul de la moyenne des réplicats 
df_list = [pd.read_csv(f'simuls/results_{f+1}.csv') for f in range(n_replicats)]
group_cols = ['n_match', 'OT_type', 'noise_level', 'overlap_rate', 'n_complex']
df_mean = pd.concat(df_list).groupby(group_cols, as_index=False).mean()


# Étape 4 : Sauvegarde pour visualisation dans R
df_mean.to_csv("simuls/results_mean.csv")
