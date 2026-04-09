---Simulation Optimal Transport---
Notre travail s'appuie sur l'hypothèse suivante : une maladie complexe correspond à une combinaison de maladies mendéliennes.
Alors, pour valider notre modèle et avoir accès à un ground_truth, nous pouvons simuler plusieurs maladies complexes à partir de mendéliennes.

> data
- profil_omim.csv.gz : contient en lignes les associations gènes-maladies monogéniques mendéliennes et en colonne les termes HPO (= phénotypes, par exemple HP:0002093 = insuffisance respiratoire). 
- profil_phecodes.csv : (pas utilisé ici) contient en ligne les phecodes et en colonne les termes HPO  

> results : stockage des fichiers générés par les scripts

> script : pour lancer les simulations et obtenir un plot, il suffit de lancer create_replicates.py puis create_graph_ot.r
