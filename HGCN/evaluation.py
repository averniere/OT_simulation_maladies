import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        q_level = np.ceil((n_scores + 1) * (1 - a)) / n_scores
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
        q_level = np.ceil((n_scores + 1) * (1 - a)) / n_scores
        seuil.append(np.quantile(scores_calib, q_level))
    return seuil


def construct_features(P, gt_set, target_k=0, seed=42):
    P_norm = P/P.sum(axis=1, keepdims=True)
    # Construction des features et de y
    top_1 = []
    ratio_12 = []
    correct = []
    concentration = []
    for i, j in gt_set:
        ranked_cols = np.argsort(P[i])[::-1]
        rank = np.where(ranked_cols == j)[0]
        if len(rank) == 0:
            continue
        top_1.append(P_norm[i, ranked_cols[0]])
        ratio_12.append(P_norm[i, ranked_cols[0]]/P_norm[i, ranked_cols[1]])
        if rank[0] <= target_k:
            correct.append(True)
        else:
            correct.append(False)
        line = P[i,:]
        concentration.append(-np.sum(line[line>0] * np.log(line[line>0])))

    X = pd.DataFrame({'top_1': top_1, 'ratio':ratio_12, 'concentration': concentration}).values
    X = (X-X.mean())/X.std()
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
    model = LogisticRegression(C=np.inf, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    metrics = evaluate_classifier(model, X_test, y_test)
    return metrics

