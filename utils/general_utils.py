import numpy as np


def pairwise_distances(x: np.ndarray, metric) -> np.ndarray:
    """ Count pairwise distances

    :param x: shape(n, d), where n - num of examples, d - num of features
    :param metric: callable function for counting metric
    :return: pairwise_dist: pairwise distances"""

    pairwise_dist = np.zeros((x.shape[0], x.shape[0]), dtype=np.float64)  # (n, n)

    for i in range(x.shape[0]):
        for j in range(x.shape[0] - i):
            dist = metric(x[i], x[i + j])
            pairwise_dist[i][i + j] = dist
            pairwise_dist[i + j][i] = dist

    return pairwise_dist


def confusion(y: np.ndarray, y_pred: np.ndarray) -> tuple:
    """ Calculate TP, FP, TR, FN

    :param y: shape(n,), gold labels
    :param y_pred: shape(n,), predicted labels
    :return: tp, fp, tn, fn: true positive, false positive, true negative, false negative"""

    tp, tn, fp, fn = 0, 0, 0, 0

    for gold_label1, predict_label1 in zip(y, y_pred):
        for gold_label2, predict_label2 in zip(y, y_pred):

            if gold_label1 == gold_label2 and predict_label1 == predict_label2:
                tp += 1
            elif gold_label1 != gold_label2 and predict_label1 == predict_label2:
                fp += 1
            elif gold_label1 == gold_label2 and predict_label1 != predict_label2:
                fn += 1
            else:
                tn += 1

    return tp, fp, tn, fn


def crosstab(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """ Calculate contingency table that displays frequency distribution

    :param y: shape(n,), gold labels
    :param y_pred: shape(n,), predicted labels
    :return: matrix: contingency table that displays frequency distribution"""

    gold_labels = np.unique(y)  # (g,)
    predict_labels = np.unique(y_pred)  # (p,)

    gold_masks = gold_labels[:, np.newaxis] == y  # (g, n)
    predict_masks = predict_labels[:, np.newaxis] == y_pred  # (p, n)

    matrix = (1 * gold_masks) @ (1 * predict_masks.T)  # (g, p)

    return matrix


def combination(n: int, k: int) -> int:
    """ Calculate the number of k-combinations from set of n elements"""

    return np.prod(np.arange(n - k + 1, n + 1)) / np.prod(np.arange(2, k + 1)) if n >= k else 0

