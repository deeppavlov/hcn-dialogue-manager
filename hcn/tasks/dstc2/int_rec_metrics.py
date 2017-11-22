from keras import backend as K
import sklearn.metrics
from parlai.core.metrics import Metrics, _exact_match, _f1_score
import numpy as np
from parlai.core.utils import round_sigfigs

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
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


class SeveralMetrics(Metrics):

    def __init__(self, opt, n_classes):
        super().__init__(opt)
        self.metrics['exact_matches_score'] = 0
        self.n_classes = n_classes

    def update(self, observations, labels):
        """Update multilabel metrics exact_matches, acc, f1"""

        with self._lock():
            self.metrics['cnt'] += 1

        # Exact match metric.
        correct = 0
        # predictions is an array of given answers!
        predictions = observations.get('text', None)
        #print('Teacher update:', predictions, labels)
        if len(predictions) != len(labels):
            exact_matches = 0
            with self._lock():
                correct = np.sum(np.array([[1. * _exact_match(predictions[i], labels[j])
                                                             for i in range(len(predictions))]
                                                            for j in range(len(labels))]))
                self.metrics['correct'] += correct
        else:
            exact_matches = np.prod(np.array([1. * _exact_match(predictions[i], labels[i])
                                              for i in range(len(predictions))]))
            with self._lock():
                correct = np.sum(np.array([1. * _exact_match(predictions[i], labels[i])
                                                            for i in range(len(predictions))]))
                self.metrics['correct'] += correct

        with self._lock():
            self.metrics['exact_matches_score'] += exact_matches

        loss = {}
        loss['correct'] = correct
        return loss

    def report(self):
        # Report the metrics over all data seen so far.
        m = {}
        m['total'] = self.metrics['cnt']
        if self.metrics['cnt'] > 0:
            m['exact_match_accuracy'] = round_sigfigs(
                self.metrics['exact_matches_score'] / self.metrics['cnt'], 4)
            m['accuracy'] = round_sigfigs(
                self.metrics['correct'] / (self.n_classes * self.metrics['cnt']), 4)
        return m

    def clear(self):
        with self._lock():
            self.metrics['cnt'] = 0
            self.metrics['correct'] = 0
            self.metrics['exact_matches_score'] = 0