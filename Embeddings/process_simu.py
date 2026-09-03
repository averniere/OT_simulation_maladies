import os
import torch
import pandas as pd
import numpy as np
import subprocess
import sys
import simulations as sim
import data

from poincare import PoincareManifold
from model import Distance_PE
from information_content import deprecated, compute_information_content

n_replicats = 3  # Nombre de simulations
result_dir = "simuls"
python_path = sys.executable

weights, diseases, all_diseases = compute_information_content(data.profils_omim, data.G_hpo_work)
ic = {t: -np.log(weights[t]) if weights.get(t, 0) > 0 else 0.0 for t in weights}

checkpoint = torch.load('logs/2026_5_7/12/model_final.pt', map_location='cpu', weights_only=False)
objects = checkpoint['objects']
hp = checkpoint['hyperparams']

manifold = PoincareManifold()
model = Distance_PE(
    n=len(objects), dim=hp['dim'],
    manifold=manifold, sparse=False, 
    learn_curvature=False, init_curvature=1., 
    weight_decay=0
    )
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

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
    results, df_truth, df_target = sim.process_simulation(
        source_data=data.profils_omim[data.hpo_cols0],
        n_complex_list=[50],  # Nombre de maladies complexes à simuler           
        n_match_list=[50, 100],  # [10 , 50], Nombre de maladies mendéliennes par maladie complexe              
        noise_levels=[0, 0.1, 0.2],  # [0, 0.05, 0.1, 0.2],   
        quantiles=list(np.arange(0.95, 0.999, 0.003)),
        epsilon=0.1,
        overlap_test=[0, 0.5],    
        group_size=20,                     
        eta_list=[1e5, 1e6],
        model=model,
        node2id=data.node2id_w,
        deprecated=deprecated,
        cost_method='hamming',  # 'wasserstein' ou 'hamming pondéré' ou 'hamming pondéré normes'
        weights_cost=data.depths,
        transp_method='classic'  # 'classic' ou 'frlc'
        )

    results.to_csv('simuls/simu_brut.csv.gz', sep=';', index=False, compression="gzip")
    df_target.to_csv("simuls/target.csv", sep=';', index=False)
    df_truth.to_csv("simuls/truth.csv", sep=';', index=False)
    
    # Appliquer add_metrics
    subprocess.run([python_path, "add_metrics.py"], check=True)
    
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
    os.remove("simuls/simu_brut.csv.gz")
    #os.remove("data/simuls/target.csv")


# Étape 3 : Calcul de la moyenne des réplicats 
df_list = [pd.read_csv(f'simuls/results_{f+1}.csv') for f in range(n_replicats)]
group_cols = ['n_match', 'OT_type', 'noise_level', 'overlap_rate', 'n_complex']
df_mean = pd.concat(df_list).groupby(group_cols, as_index=False).mean()


# Étape 4 : Sauvegarde pour visualisation dans R
df_mean.to_csv("simuls/results_mean.csv")
