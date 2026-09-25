import pandas as pd


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
                    
    return df
