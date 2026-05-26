import pandas as pd
import numpy as np

quantile_list = list(np.arange(0.95, 0.999, 0.003))

simu = pd.read_csv('simuls/simu_brut.csv.gz', sep = ';')

def add_recall_precision(df, quantiles):
    for quantile in quantiles:
        # Colonnes à traiter
        quantile_col = f"Associations_quantile_{quantile}"
        recall_col = f"Recall_quantile_{quantile}"
        precision_col = f"Precision_quantile_{quantile}"
        #true_pos_col = f"True_pos_{quantile}"
        #liste_simulee_col = f"Liste_simu_quantile_{quantile}"
        
        df[recall_col] = 0.0
        df[precision_col] = 0.0
        #df[true_pos_col] = 0
        #df[liste_simulee_col] = 0
        df[quantile_col] = df[quantile_col].astype(str)
        # Traiter chaque ligne du dataframe
        for idx, row in df.iterrows():

            if pd.notna(row[quantile_col]) and pd.notna(row["Mendelian_Sources"]):
                # Transformer les chaînes en ensembles
                quantile_list = set(map(str.strip, row[quantile_col].split(",")))
                reference_list = set(map(str.strip, row["Mendelian_Sources"].split(",")))
                
                # Calculer les métriques
                true_positives = len(quantile_list & reference_list)  # Intersection
                recall = true_positives / len(reference_list) if len(reference_list) > 0 else 0
                precision = true_positives / len(quantile_list) if len(quantile_list) > 0 else 0
                
                df.at[idx, recall_col] = recall
                df.at[idx, precision_col] = precision
                #df.at[idx, true_pos_col] = true_positives
                #df.at[idx, liste_simulee_col] = len(quantile_list)
    
    return df 

add_recall_precision(simu, quantile_list)


columns_to_drop = [f'Associations_quantile_{q}' for q in quantile_list]

recall_prec = simu.drop(columns = ['Complex_Disease', 'Mendelian_Sources'] + columns_to_drop)

#recall_prec.to_csv('Database/recallprectest.csv')


# Colonnes sur lesquelles regrouper
group_cols = ['n_match', 'OT_type', 'noise_level', 'overlap_rate', 'n_complex', 'eta']

# Calcul des moyennes
result = recall_prec.groupby(group_cols, as_index=False).mean()

result.to_csv('simuls/simul.csv')
