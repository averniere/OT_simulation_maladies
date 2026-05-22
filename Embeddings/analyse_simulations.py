#Import
import torch
import pandas as pd
import numpy as np
import networkx as nx
import re 

from tqdm import tqdm
from ot.optim import gcg
from scipy.sparse import csgraph
from poincare import PoincareManifold
from model import Distance_PE
from information_content import deprecated
from data_utils import *
from OT_utils import *


def simulate_disease(df_mendelien, nb_complex, nb_per_complex, group_size, overlap_rate):

    # Liste des noms des maladies complexes
    complex_names = []
    for i in range(1, nb_complex + 1):
        complex_names.append(f"Complexe_{i}")

    # Dictionnaire : pour chaque maladie complexe, la liste des maladies mendéliennes associées
    complex_mendelian_dict = {}
    for name in complex_names:
        complex_mendelian_dict[name] = []

    # Répartitions des complexes en groupes
    groups = []
    for i in range(0, nb_complex, group_size):
        groupe = complex_names[i : i + group_size]  #['Complexe_1', 'Complexe_2'] si group_size = 2
        groups.append(groupe)

    mendelian_list = list(df_mendelien.index.astype(int))

    used_mendelian = set()  # mendéliennes déjà utilisées dans les groupes précédents

    # On traite les complexes groupe par groupe
    for group in groups:
        total_slots = group_size * nb_per_complex              # nombre total de slots dans le groupe
        nb_shared_slots = int(overlap_rate * total_slots)      # slots occupés par des mendéliennes partagées
        nb_shared_mendelian = nb_shared_slots // 2             # chaque mendélienne partagée occupe 2 slots
        nb_unique_slots = total_slots - nb_shared_slots        # slots occupés par des mendéliennes uniques

        # Mendéliennes pas encore utilisées
        remaining = []
        for m in mendelian_list:
            if m not in used_mendelian:
                remaining.append(m)

        if len(remaining) < nb_shared_mendelian + nb_unique_slots:
            raise ValueError("Pas assez de maladies mendéliennes disponibles.")

        # On pioche les mendéliennes partagées
        shared = np.random.choice(remaining, size=nb_shared_mendelian, replace=False).tolist()
        for m in shared:
            used_mendelian.add(m)

        # On pioche les mendéliennes uniques parmi celles qui restent
        remaining_after_shared = []
        for m in mendelian_list:
            if m not in used_mendelian:
                remaining_after_shared.append(m)
        unique = np.random.choice(remaining_after_shared, size=nb_unique_slots, replace=False).tolist()
        for m in unique:
            used_mendelian.add(m)

        # Distribution dans les complexes du groupe
        """
        On doit assigner à chaque complexe exactement nb_per_complex mendéliennes, 
        les mendéliennes partagées doivent apparaître dans exactement 2 complexes.
        => shared apparaissent 2 fois, unique 1 fois.
        """
        slots_a_distribuer = []
        for m in shared:
            slots_a_distribuer.append(m)
            slots_a_distribuer.append(m)
        for m in unique:
            slots_a_distribuer.append(m)

        # On mélange les slots et on les distribue un par un aux complexes
        np.random.shuffle(slots_a_distribuer)

        slot_index = 0
        for complexe in group:
            assigned = []

            while len(assigned) < nb_per_complex:
                m = slots_a_distribuer[slot_index % len(slots_a_distribuer)]
                slot_index += 1

                if m not in assigned:
                    assigned.append(m)

            complex_mendelian_dict[complexe] = assigned

    # Construction des DataFrames
    complex_profiles = []
    complex_groundtruth = []

    for name in complex_names:
        mendelian_sources = complex_mendelian_dict[name]
        sous_tableau = df_mendelien.loc[mendelian_sources]
        somme_par_hp = sous_tableau.sum(axis=0)
        profile = (somme_par_hp > 0).astype(int)

        complex_profiles.append(profile)
        complex_groundtruth.append(",".join(str(x) for x in mendelian_sources))

    df_complex = pd.DataFrame(complex_profiles, index=complex_names)
    df_truth = pd.DataFrame({"Complex_Disease": complex_names,"Mendelian_Sources": complex_groundtruth})

    return df_complex, df_truth


# Regularized Optimal transport
def Ot_Laplacienne(a, b, xs, xt, M, S, epsilon, eta, numItermax=100, stopThr=1e-9, numInnerItermax=100000,stopInnerThr=1e-9, log=False, verbose=False):

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
        val_lap = np.trace(G.T.dot(lS_sym).dot(G))
        # si on considere similarité dans la cible egalement avec un param alphe ici = 0.5
        # val_lap = 0.5 * np.trace(G.T.dot(lS2).dot(G)) + 0.5 * np.trace(G.dot(lc2).dot(G.T))
        return val_lap

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

# Add noise to matrix
def add_noise_to_matrix(matrix, noise_level):
    """ 
    Exemple : noise_level = 0.1
        Si 100 "1" dans la matrice, on en prend 10 et on les change en 0
        Si 300 "0" dans la matrice, on en prend 30 et on les change en 1
    """
    if noise_level == 0:
        return matrix.copy()

    noisy_array = matrix.to_numpy().copy()

    # Indices des 1 et des 0
    ones_indices = np.argwhere(noisy_array == 1)
    zeros_indices = np.argwhere(noisy_array == 0)

    # Nombre d'éléments à modifier dans chaque sens
    num_to_remove = int(noise_level * len(ones_indices))  # 1 → 0
    num_to_add = int(noise_level * len(zeros_indices))    # 0 → 1

    # Sélection aléatoire des indices à modifier
    indices_to_remove = ones_indices[np.random.choice(len(ones_indices), num_to_remove, replace=False)]
    indices_to_add = zeros_indices[np.random.choice(len(zeros_indices), num_to_add, replace=False)]

    # Application du bruit
    for i, j in indices_to_remove:
        noisy_array[i, j] = 0

    for i, j in indices_to_add:
        noisy_array[i, j] = 1

    return pd.DataFrame(noisy_array, index=matrix.index, columns=matrix.columns)

# Filter by quantiles
def filter_by_quantile(matrix_df, quantiles, n_match, OT_type, noise_level, overlap_rate, n_complex):
    result_df = pd.DataFrame()
    
    for quantile in quantiles:
        threshold = np.quantile(matrix_df.values.flatten(), quantile)   # Calcul du seuil pour le quantile
        filtered_matrix = matrix_df.where(matrix_df > threshold)  # Filtrer les cellules qui dépassent le seuil
        
        # Création des listes de diseases mendéliennes associées pour chaque disease complexe
        assoc_dict = {}

        for complex_disease in filtered_matrix.columns:   # on parcourt chaque complexe (Complexe_1, Complexe_2...)
            colonne = filtered_matrix[complex_disease]
            masque = colonne.notna()  # On garde uniquement les mendéliennes qui ont un score (pas NaN = elles ont passé le seuil)
            mendéliennes_selectionnees = filtered_matrix.index[masque]

            # On convertit en liste de strings et on joint avec des virgules
            mendéliennes_str = mendéliennes_selectionnees.astype(str)
            resultat = ",".join(mendéliennes_str)
            
            # On stocke dans le dictionnaire
            assoc_dict[complex_disease] = resultat
        
        # Transformer en DataFrame pour ce quantile
        quantile_df = pd.DataFrame({"Complex_Disease": assoc_dict.keys(), f"Associations_quantile_{quantile}": assoc_dict.values()})
        
        if result_df.empty:
            result_df = quantile_df
        else:
            result_df = pd.merge(result_df, quantile_df, on="Complex_Disease", how="outer")        
    # Ajouter les colonnes 'n_match', 'OT_type', 'Noise_Level'
    result_df["n_match"] = n_match
    result_df["OT_type"] = OT_type
    result_df["noise_level"] = noise_level
    result_df["overlap_rate"] = overlap_rate
    result_df["n_complex"] = n_complex
    
    return result_df

# Create S
def similarity_matrix(df, mendelian_list):
    maladies_mendeliennes_df = set()
    for sublist in df['Mendelian_Sources'].str.split(','):   # pour chaque ligne, on split par virgule
        for m in sublist:                                    # pour chaque mendélienne dans la liste
            maladies_mendeliennes_df.add(int(m.strip()))
    maladies_mendeliennes_totale = list(sorted(
        maladies_mendeliennes_df.union({int(m) for m in mendelian_list})))  # Ajouter des maladies supplémentaires
    n = len(maladies_mendeliennes_totale)

    # Création d'un index pour les maladies mendéliennes
    mendeliennes_index = {m: i for i, m in enumerate(maladies_mendeliennes_totale)}

    # Matrice binaire initialisée à 0
    matrice_binaire = np.zeros((n, n), dtype=float)

    # Remplissage de la matrice
    for maladies in df["Mendelian_Sources"]:
        maladies_associees = [int(m.strip()) for m in maladies.split(',')] 
        indices = [mendeliennes_index[m] for m in maladies_associees]
        #for i in indices:
            #for j in indices:
                #matrice_binaire[i, j] = 1
        ix = np.ix_(indices, indices)
        matrice_binaire[ix] = 1
    for i in range(n):
        matrice_binaire[i, i] = 1.0
    return (matrice_binaire, maladies_mendeliennes_totale)

# Random S
def dissimilarity_matrix(similarity):
    #Compter le nombre total de 1
    num_ones = int(np.sum(similarity))
    rows, cols = similarity.shape

    #Calculer les paires symétriques
    indices = [(i, j) for i in range(rows) for j in range(i + 1, cols)]
    np.random.shuffle(indices)

    #Sélectionner les paires pour respecter num_ones (en retirant la diagonale)
    num_pairs = (num_ones - rows) // 2
    selected_pairs = indices[:num_pairs]

    #Construire la matrice symétrique
    symmetric_matrix = np.zeros((rows, cols), dtype=float)
    
    #Ajouter les paires symétriques
    for i, j in selected_pairs:
        symmetric_matrix[i, j] = 1
        symmetric_matrix[j, i] = 1

    #Ajouter les 1 sur la diagonale
    for i in range(rows):
        symmetric_matrix[i, i] = 1.0

    return symmetric_matrix

# Filtrer maladies
def filtrer_maladies(df, matrice, colonne_maladies, all_disease = False):
    # Pour ne garder que les mendéliennes réellement utilisées dans la simulation.
    if all_disease:
        return (matrice)
    else :
        # Extraire les listes de maladies, les décomposer, et les mettre dans un ensemble pour éviter les doublons
        maladies = set(
            maladie.strip()
            for liste in df[colonne_maladies].dropna()  # Ignorer les valeurs NaN
            for maladie in liste.split(",")
        )

        # Filtrer la matrice avec les maladies trouvées
        #maladies = list(maladies)  # Convertir en liste pour l'utilisation avec les index
        maladies = {int(m.strip()) for liste in df[colonne_maladies].dropna() for m in liste.split(",")}
        matrice_filtre = matrice.loc[
            matrice.index.intersection(maladies)]
        return matrice_filtre


def process_simulation(
    source_data, 
    n_complex_list, 
    n_match_list, 
    noise_levels, 
    quantiles, 
    epsilon, 
    overlap_test, 
    group_size, 
    eta_list,
    model,
    node2id,
    deprecated
    ):

    #global_result = pd.DataFrame()
    all_results = []
    all_truths = []
    #global_truth = pd.DataFrame()

    for overlap_rate in overlap_test:
        for n_complex in n_complex_list:
            for n_match in tqdm(n_match_list, desc="Testing different n_match values"):

                # Simulation des maladies complexes
                simulation = simulate_disease(source_data, n_complex, n_match, group_size, overlap_rate)
                target_data = simulation[0]
                df_truth = simulation[1]

                # Filtrer les mendéliennes utilisées dans la simulation
                source_data_filtre = filtrer_maladies(df_truth, source_data, 'Mendelian_Sources')
                mendelian_list = source_data_filtre.index

                # Créer la matrice de similarité S
                df_simi = df_truth.copy()
                similarity = similarity_matrix(df_simi, mendelian_list)
                source_data_filtre = source_data_filtre.reindex(similarity[1])

                complementary_matrices = {
                    "similarity": similarity[0],
                    "dissimilarity": dissimilarity_matrix(similarity[0]),
                }

                # Stocker ground truth
                df_truth['n_match'] = n_match
                df_truth['overlap_rate'] = overlap_rate
                df_truth['n_complex'] = n_complex
                #global_truth = pd.concat([global_truth, df_truth], ignore_index=True)
                all_truths.append(df_truth)

                for noise_level in noise_levels:

                    # Bruit sur les profils complexes
                    noisy_matrix = add_noise_to_matrix(target_data, noise_level)
            
                    # Calcul des distances
                    print("Source data filtré")
                    print(source_data_filtre.shape)
                    print(noisy_matrix.shape)
                    cost_matrix = compute_costs_matrix_wasserstein3(source_data_filtre, noisy_matrix, node2id, model, deprecated)
                    print("Cost matrix :", cost_matrix.shape)
                    print(f"Mean = {np.mean(cost_matrix)}")
                    print(f"Min = {np.min(cost_matrix)}")
                    print(f"Max = {np.max(cost_matrix)}")
                    print(f"Median = {np.median(cost_matrix)}")
                    cost_matrix_df = pd.DataFrame(cost_matrix, index=source_data_filtre.index, columns=target_data.index)
                    
                    # Raw distance
                    OT_type = "Raw"
                    jaccard_distances_results = filter_by_quantile(cost_matrix_df, quantiles, n_match, OT_type, noise_level, overlap_rate, n_complex)
                    #global_result = pd.concat([global_result, jaccard_distances_results], ignore_index=True)
                    all_results.append(jaccard_distances_results)

                    # Distributions de poids uniformes
                    a = np.ones(len(source_data_filtre.index)) / len(source_data_filtre.index)
                    b = np.ones(len(target_data.index)) / len(target_data.index)

                    # Optimal transport
                    print("Compute transport...")
                    epsilon0 = epsilon*np.mean(cost_matrix)
                    ot_plan, ot_cost = compute_transport_sinkhorn(cost_matrix, None, None, epsilon0, 10000, 1e-4, False)
                    print("Finished !")
                    transport_matrix_df = pd.DataFrame(ot_plan, index=source_data_filtre.index, columns=target_data.index)
                    OT_type = "OT"
                    ot_results = filter_by_quantile(transport_matrix_df, quantiles, n_match, OT_type, noise_level, overlap_rate, n_complex)
                    #global_result = pd.concat([global_result, ot_results], ignore_index=True)
                    all_results.append(ot_results)

                    # OT Laplacien
                    Xs_real = source_data_filtre.to_numpy()
                    Xt_simu = target_data.to_numpy()

                    for eta in eta_list:
                        for S_name, S_value in complementary_matrices.items():

                            gamma_opt = Ot_Laplacienne(a, b, xs=Xs_real, xt=Xt_simu, M=cost_matrix, S=S_value, epsilon=epsilon0, eta=eta)
                            gamma_opt_df = pd.DataFrame(gamma_opt, index=source_data_filtre.index, columns=target_data.index)
                            OT_type = f"OT regularized, {S_name}"
                            laplace_results = filter_by_quantile(gamma_opt_df, quantiles, n_match, OT_type, noise_level, overlap_rate, n_complex)
                            #global_result = pd.concat([global_result, laplace_results], ignore_index=True)
                            all_results.append(laplace_results)

    global_result = pd.concat(all_results, ignore_index=True)
    global_truth = pd.concat(all_truths, ignore_index=True)
    # Avec la vérité terrain
    global_result = global_result.merge(
        global_truth[['Complex_Disease', 'n_match', 'Mendelian_Sources', 'overlap_rate']],
        on=['Complex_Disease', 'n_match', 'overlap_rate'],
        how='left'
    )

    return global_result, df_truth, target_data


# Débuggage du code 
hp_ids = []
parents_list = []

with open("../data/HPOs.csv", "r") as f:
    next(f)
    for line in f:
        hp_id = line.split(';')[0]
        
        # Extraire uniquement la liste contenant des IDs HP:XXXXXXX
        match = re.search(r"\[([^\]]*'HP:\d{7}'[^\]]*)\]", line)
        if match:
            parents = re.findall(r"HP:\d{7}", match.group(0))
        else:
            parents = []
        
        hp_ids.append(hp_id)
        parents_list.append(parents)

df_hpo = pd.DataFrame({'hp_id': hp_ids, 'parents': parents_list})

G_hpo_work = nx.DiGraph()
for hp_id in hp_ids:
    G_hpo_work.add_node(hp_id)
for hp_id, parents in zip(hp_ids, parents_list):
    for parent_id in parents:
        if parent_id in G_hpo_work:
            G_hpo_work.add_edge(hp_id, parent_id)

objects_w = list(G_hpo_work.nodes())
node2id_w = {n: i for i, n in enumerate(objects_w)}


def read_hpoa(path):
    with open(path, 'r') as f:
        skip = sum(1 for line in f if line.startswith('#'))
    return pd.read_csv(path, sep='\t', skiprows=skip, low_memory=False)


df_hpoa = read_hpoa('../data/phenotype_omim_orpha.hpoa')
df_hpoa['disease_name'] = df_hpoa['disease_name'].str.lower().str.strip().str.replace(r'[\s\-]+', ' ', regex=True)
df_hpoa.tail()

correspondence_exacte = build_disease_correspondence(df_hpoa)
print(f"Correspondances trouvées : {len(correspondence_exacte)}")

# Construction de deux dataframes à partir de df_hpoa
df_pivot = df_hpoa[['database_id', 'hpo_id']].drop_duplicates()
df_pivot['values']=1.
df_pivot = pd.pivot_table(data=df_pivot, values='values', index='database_id', columns='hpo_id', aggfunc='max', fill_value=0)
df_pivot.columns.name = None
df_pivot = df_pivot.reset_index()

df_orpha = df_pivot[df_pivot['database_id'].str.startswith('ORPHA:')]
df_orpha = df_orpha[df_orpha['database_id'].isin(correspondence_exacte['orpha_id'])]

df_omim = df_pivot[df_pivot['database_id'].str.startswith('OMIM:')]
df_omim = df_omim[df_omim['database_id'].isin(correspondence_exacte['omim_id'])]

hpo_cols = [c for c in df_omim.columns if c.startswith('HP:')]

profils_omim = pd.read_csv("../data/profils_omim.csv.gz", index_col=0)
profils_omim = profils_omim.reset_index()
hpo_cols0 = [c for c in profils_omim.columns if c.startswith('HP:')]

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

results, df_truth, df_target = process_simulation(
    source_data=profils_omim[hpo_cols0],
    n_complex_list=[50],               
    n_match_list=[10],  # , 50],              
    noise_levels=[0, 0.05, 0.1, 0.2],   
    quantiles=list(np.arange(0.95, 0.999, 0.003)),
    epsilon=0.1,
    overlap_test=[0, 0.6],    
    group_size=10,                     
    eta_list=[1000],
    model=model,
    node2id=node2id_w,
    deprecated=deprecated
)

results.to_csv('simuls/simu_brut.csv.gz', sep=';', index=False, compression="gzip")
df_target.to_csv("simuls/target.csv", sep=';', index=False)
df_truth.to_csv("simuls/truth.csv", sep=';', index=False)