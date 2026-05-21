import os
import pandas as pd
import subprocess
import sys

n_replicats = 5  # Nombre de simulations
result_dir = "results"
#python_path = r"C:\Users\Jules\AppData\Local\Programs\Python\Python312\python.exe"
python_path = sys.executable

# S'assurer que le dossier existe
os.makedirs(result_dir, exist_ok=True)

if os.path.exists("results/results_to_r.csv"):
    os.remove("results/results_to_r.csv")
if os.path.exists("results/simu_brut.csv.gz"):
    os.remove("results/simu_brut.csv.gz")

for k in range(n_replicats):
    if os.path.exists(f"results/results_{k+1}.csv"):
        os.remove(f"results/results_{k+1}.csv")

# Étape 1 et 2 : Exécuter les simulations et stocker les résultats
for i in range(n_replicats):
    print(f"Lancement de la simulation {i+1}/{n_replicats}")

    # Lancer la simulation
    subprocess.run([python_path, "process_simu.py"], check=True)
    
    # Appliquer add_metrics
    subprocess.run([python_path, "add_metrics.py"], check=True)
    
    # Renommer et stocker le fichier resultats.csv
    result_file = os.path.join(result_dir, f"results_{i+1}.csv")
    os.rename("results/simul.csv",result_file)

    # renommer et stocker les fichiers target
    target_file = os.path.join("../data/simuls", f"target_{i+1}.csv")
    os.rename("../data/simuls/target.csv", target_file)
    
    # Supprimer le fichier volumineux de base
    os.remove("results/simu_brut.csv.gz")
    #os.remove("data/simuls/target.csv")


# Étape 3 : Calcul de la moyenne des réplicats 
df_list = [pd.read_csv(f'results/results_{f+1}.csv') for f in range(n_replicats)]
group_cols = ['n_match', 'OT_type', 'noise_level', 'overlap_rate', 'n_complex']
df_mean = pd.concat(df_list).groupby(group_cols, as_index=False).mean()


# Étape 4 : Sauvegarde pour visualisation dans R
df_mean.to_csv("results/results_mean.csv")
