import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_score, recall_score


def seuil_masse(P, gt_set, alphas=[0.10, 0.05], seed=42):
    np.random.seed(seed)
    P_norm = P/np.sum(P, axis=1, keepdims=True)
    n = len(gt_set)

    indices = np.random.permutation(n)
    n_calib = int(0.2 * n)
    calib_indices = indices[:n_calib]
    test_indices = indices[n_calib:]

    paires_calibration = [list(gt_set)[i] for i in calib_indices]
    print(f"Taille du jeu de calibration : {len(paires_calibration)}")
    paires_test = [list(gt_set)[i] for i in test_indices]
    print(f"Taille du jeu de test : {len(paires_test)}")

    scores_calib = []
    for i, j in paires_calibration:
        scores_calib.append(1-P_norm[i, j])

    n_scores = len(scores_calib)
    seuil = []
    for a in alphas:
        q_level = np.ceil((n + 1) * (1 - a)) / n_scores
        seuil.append(np.quantile(scores_calib, q_level))
    return seuil


def seuil_rang(P, gt_set, alphas=[0.1, 0.05], seed=42):
    np.random.seed(seed)
    P_norm = P/np.sum(P, axis=1, keepdims=True)
    n = len(gt_set)
    m = P_norm.shape[1]

    indices = np.random.permutation(n)
    n_calib = int(0.2 * n)
    calib_indices = indices[:n_calib]
    test_indices = indices[n_calib:]

    paires_calibration = [list(gt_set)[i] for i in calib_indices]
    print(f"Taille du jeu de calibration : {len(paires_calibration)}")
    paires_test = [list(gt_set)[i] for i in test_indices]
    print(f"Taille du jeu de test : {len(paires_test)}")

    scores_calib = []
    for i, j in paires_calibration:
        ranked_cols = np.argsort(P[i])[::-1]
        rank = np.where(ranked_cols == j)[0]
        if len(rank) == 0:
            continue
        rank = rank[0] + 1
        scores_calib.append(rank)

    n_scores = len(scores_calib)
    seuil = []
    for a in alphas:
        q_level = np.ceil((n + 1) * (1 - a)) / n_scores
        seuil.append(np.quantile(scores_calib, q_level))
    return seuil


def construct_features(P, gt_set, seed=42):
    P_norm = P/P.sum(axis=1, keepdims=True)
    # Construction des features et de y
    top_1 = []
    ratio_12 = []
    rank_true = []
    correct = []
    concentration = []
    for i, j in gt_set:
        ranked_cols = np.argsort(P[i])[::-1]
        rank = np.where(ranked_cols == j)[0]
        if len(rank) == 0:
            continue
        rank_true.append(rank[0] + 1)
        top_1.append(P_norm[i, ranked_cols[0]])
        ratio_12.append(P_norm[i, ranked_cols[0]]/P_norm[i, ranked_cols[1]])
        if rank[0] == 0:
            correct.append(True)
        else:
            correct.append(False)
        line = P[i,:]
        concentration.append(-np.sum(line[line>0] * np.log(line[line>0])))

    X = pd.DataFrame({'top_1': top_1, 'ratio':ratio_12, 'rank':rank_true}).values
    X=(X-X.mean())/X.std()
    y = correct

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    print(f"Train : {len(X_train)}\n Test : {len(X_test)}")
    return X_train, X_test, y_train, y_test


def evaluate_classifier(model, X_test, y_test, average="weighted", show_confusion=True):
    """
    Évalue un modèle de classification sklearn (binaire ou multiclasses).
    Retourne un dictionnaire contenant : la prédiction de la classe, la prediction de la probabilité P(Y=1|X), l'accuracy, la précision et le recall.
    Si show_confusion=True : affiche la matrice de confusion.
    """

    y_pred = model.predict(X_test)
    y_pred_proba = None if hasattr(model, "predict_proba") else model.predict_proba(X_test)

    # Métriques
    accuracy = model.score(X_test, y_test)
    precision = precision_score(y_test, y_pred, average=average)
    recall = recall_score(y_test, y_pred, average=average)

    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")

    # Matrice de confusion
    if show_confusion:
        labels = model.classes_ if hasattr(model, "classes_") else None
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=labels
        )
        disp.plot()
        plt.grid(False)
        plt.show()

    return {
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm if show_confusion else None
    }


def logistic_reg(P, gt_set, seed=42):
    X_train, X_test, y_train, y_test = construct_features(P, gt_set, seed)
    model = LogisticRegression(penalty=None, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    metrics = evaluate_classifier(model, X_test, y_test)
    return metrics


def seuil_commun(df1, df2, P, gt_set, rg=2):
    '''
    Trace g(x)=P(Y=1|X>=x) où Y Bernoulli égale à 1 si le match est dans le Top rg,
    et X indique le nombre de termes en commun entre les deux maladies de la paire.
    '''
    def wilson_ci(count, n, confidence=0.95):
        """Retourne (p_hat, borne_basse, borne_haute) via l'intervalle de Wilson."""
        if n == 0:
            return np.nan, np.nan, np.nan
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        p_hat = count / n
        denom = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denom
        margin = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
        return p_hat, center - margin, center + margin

    def communs(df1, df2):
        n=len(df1)
        m=len(df2)
        hpo_cols = [c for c in df1.columns if c.startswith('HP:')]
        X, Y = df1[hpo_cols].values, df2[hpo_cols].values
        C = (X @ Y.T)/np.sum(X, axis=1, keepdims=True)  # Part de termes en communs par rapport à OMIM
        return C

    def g(x, C, P, gt_set, k=rg):
        mask = C >= x
        #denom = mask.sum()/(n*m)  # Si l'on calcule la probabilité sur l'ensemble des maladies et pas seulement sur les paires
        count = 0
        denom = 0
        for i, j in gt_set:
            if not mask[i, j]:
                continue
            denom += 1
            ranked_cols = np.argsort(P[i])[::-1]
            rank = np.where(ranked_cols == j)[0]
            if len(rank) == 0:
                continue
            if mask[i, j] and rank[0] <= k :
                count += 1
        # count/=len(gt_set)
        if denom == 0:
            return np.nan
        return count, denom

    C = communs(df1, df2)
    x_values = np.linspace(0, 1, 50)
    results = [g(x, C, P, gt_set) for x in x_values]
    counts = [r[0] for r in results]
    denoms = [r[1] for r in results]

    ci_results = [wilson_ci(c, n) for c, n in zip(counts, denoms)]
    g_values = [r[0] for r in ci_results]
    lower_bounds = [r[1] for r in ci_results]
    upper_bounds = [r[2] for r in ci_results]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(x_values, g_values, marker='o', markersize=3, color='tab:blue', label='g(x)')
    ax1.fill_between(x_values, lower_bounds, upper_bounds, color='tab:blue', alpha=0.2, label='IC 95% (Wilson)')
    ax1.axhline(y=0.8, color='red', linestyle='--', label='y = 0.95')
    ax1.set_xlabel("x")
    ax1.set_ylabel("g(x)")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='lower left')

    ax2 = ax1.twinx()
    ax2.plot(x_values, denoms, color='tab:gray', alpha=0.4, linestyle=':', label='n (taille échantillon)')
    ax2.set_ylabel("Nombre de paires évaluées", color='tab:gray')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()