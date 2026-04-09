import numpy as np

eta_list = [1000]

quantile_list = list(np.arange(0.95, 0.999, 0.003))


import csv

# Écriture dans un fichier CSV
with open("script/eta_liste.csv", "w", newline="") as fichier_csv:
    writer = csv.writer(fichier_csv)
    writer.writerow(eta_list)

