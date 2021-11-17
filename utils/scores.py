import numpy as np

import general_utils
import metrics


def silhouette_score(x: np.ndarray, y: np.ndarray, metric) -> float:
    """ Assess the quality of clustering using Silhouette Score

    :param x: shape(n, d), where n - num of examples, d - num of features
    :param y: shape(n,)
    :param metric: callable function for counting metric"""

    pairwise_dist = general_utils.pairwise_distances(x, metric)

    score = 0
    labels = np.unique(y)  # (k,), get unique labels
    masks = labels[:, np.newaxis] == y  # (k, n), whether example belong to cluster
    counts = np.sum(masks, axis=1)  # (k,), num of examples in clusters

    for ind1, (mask, count) in enumerate(zip(masks, counts)):  # iterate clusters

        # (for every example in the cluster) mean dist from one example to others from the same cluster
        a = np.sum(pairwise_dist[mask][:, mask], axis=1) / (count - 1) # (count,)

        # (for every example in the cluster) mean dist from one example to all examples from the nearest other cluster
        b = np.full((count,), np.inf)  # (count,)
        for ind2, other_mask in enumerate(masks):  # iterate other clusters
            if ind2 == ind1:
                continue

            # (for every example in the cluster) mean dist from one example to all examples from other cluster
            temp = np.sum(pairwise_dist[mask][:, other_mask], axis=1) / counts[ind2]  # (count,)
            b = np.min(np.vstack([b, temp]), axis=0)

        score += np.sum((b - a) / np.max(np.vstack([a, b]), axis=0))

    return score / x.shape[0]


def calinski_harabasz_score(x: np.ndarray, y: np.ndarray) -> float:
    """ Assess the quality of clustering using Calinski-Harabasz Index

    :param x: shape(n, d), where n - num of examples, d - num of features
    :param y: shape(n,)"""

    labels = np.unique(y)  # (k,), get unique labels
    masks = labels[:, np.newaxis] == y  # (k, n), whether example belong to cluster
    counts = np.sum(masks, axis=1)  # (k,), num of examples in clusters

    cluster_centoids = masks @ x / counts[:, np.newaxis]  # (k, d)
    global_centroid = np.sum(x, axis=0) / x.shape[0]  # (d,)

    score = 0
    for count, cc in zip(counts, cluster_centoids):
        score += metrics.l2(cc, global_centroid) * count

    temp = 0
    for mask, cc in zip(masks, cluster_centoids):
        examples = x[mask]
        for ex in examples:
            temp += metrics.l2(ex, cc)

    score = score / temp * (x.shape[0] - labels.shape[0]) / (labels.shape[0] - 1)
    return score


def rand_score(y: np.ndarray, y_pred: np.ndarray) -> float:
    """ Assess the quality of clustering using Rand Index

    :param y: shape(n,), gold labels
    :param y_pred: shape(n,), predicted labels"""

    tp, fp, tn, fn = general_utils.confusion(y, y_pred)
    return (tp + tn) / (tp + tn + fp + fn)


def jaccard_score(y: np.ndarray, y_pred: np.ndarray):
    """ Assess the quality of clustering using Jaccard Index

    :param y: shape(n,), gold labels
    :param y_pred: shape(n,), predicted labels"""

    tp, fp, tn, fn = general_utils.confusion(y, y_pred)
    return tp / (tp + fn + fp)


def adjusted_rand_score(y: np.ndarray, y_pred: np.ndarray) -> float:
    """ Assess the quality of clustering using Adjusted Rand Index

    :param y: shape(n,), gold labels
    :param y_pred: shape(n,), predicted labels"""

    matrix = general_utils.crosstab(y, y_pred)
    n = y.shape[0]

    a = np.sum(matrix, axis=1)
    b = np.sum(matrix, axis=0)

    temp1 = sum(general_utils.combination(ai, 2) for ai in a)
    temp2 = sum(general_utils.combination(bi, 2) for bi in b)

    index = sum(general_utils.combination(nij, 2) for ni in matrix for nij in ni)
    expected_index = temp1 * temp2 / general_utils.combination(n, 2)
    max_index = (temp1 + temp2) / 2

    return (index - expected_index) / (max_index - expected_index)