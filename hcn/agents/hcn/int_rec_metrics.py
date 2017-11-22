import sklearn.metrics
import numpy as np

EPS = 1e-16

def precision(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + EPS)
    return precision

def recall(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + EPS)
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    if np.sum(np.round(np.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + EPS)
    return fbeta_score

def fmeasure(y_true, y_pred, average=None):
    return fbeta_score(y_true, y_pred, beta=1)

def roc_auc_score(y_true, y_pred, average='macro'):
    """Compute Area Under the Curve (AUC) from prediction scores.

    Args:
        y_true: true binary labels
        y_pred: target scores, can either be probability estimates of the positive class

    Returns:
        Area Under the Curve (AUC) from prediction scores
    """
    try:
        return sklearn.metrics.roc_auc_score(y_true, y_pred, average=average)
    except ValueError:
        return 0.

def precision_recall_fscore_support(y_true, y_pred, average='macro'):

    try:
        return sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average=average)
    except ValueError:
        return 0.