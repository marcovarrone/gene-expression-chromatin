import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np


def evaluate_embedding(X_train, y_train, classifier_name, n_iter=10, seed=42, verbose=1):
    rocs = []
    accuracies = []
    f1s = []
    precisions = []
    recalls = []
    for i in range(n_iter):
        skf = StratifiedKFold(n_splits=10,shuffle=True)
        for train_index, val_index in skf.split(X_train, y_train):
            if classifier_name == 'mlp':
                classifier = MLPClassifier(max_iter=500, hidden_layer_sizes=(800,))
            elif classifier_name == 'svm':
                classifier = SVC(gamma='scale')
            elif classifier_name == 'mlp_2':
                classifier = MLPClassifier(max_iter=500, hidden_layer_sizes=(100, 100,), random_state=seed)
            else:
                classifier = LogisticRegression(max_iter=500, solver='lbfgs')

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train[train_index], y_train[train_index])
            X_valid_scaled = scaler.transform(X_train[val_index])

            classifier.fit(X_train_scaled, y_train[train_index])
            y_pred = classifier.predict(X_valid_scaled)

            roc = roc_auc_score(y_train[val_index], y_pred)
            accuracy = classifier.score(X_valid_scaled, y_train[val_index])
            f1 = f1_score(y_train[val_index], y_pred)
            precision = precision_score(y_train[val_index], y_pred)
            recall = recall_score(y_train[val_index], y_pred)
            rocs.append(roc)
            accuracies.append(accuracy)
            f1s.append(f1)
            precisions.append(precision)
            recalls.append(recall)

            if verbose:
                print("Accuracy:", accuracy, "- ROC:", roc, "- F1:", f1, "- Precision:", precision, "- Recall:", recall)
    return {'acc': accuracies, 'roc': rocs, 'f1': f1s, 'precision': precisions, 'recall': recalls}


def set_gpu(active=False):
    if not active:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

def set_n_threads(n_threads):
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
