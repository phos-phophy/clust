import numpy as np

import general_utils


def agglomerative(x: np.ndarray, nc: int, metric) -> np.ndarray:
    """ Clustering using agglomerative method with distance between centroids

    :param x: shape(n, d), where n - num of examples, d - num of features
    :param nc: number of clusters
    :param metric: callable function for counting metric
    :return: y: shape(n,), predict labels"""

    cur_nc = x.shape[0]

    centroids = x  # (cur_nc, d) - "one example = one cluster"
    y = np.arange(x.shape[0])  # (n,)
    labels = np.unique(y)  # (cur_nc,), get unique labels
    masks = labels[:, np.newaxis] == y  # (cur_nc, n), whether example belong to cluster

    # distance between cluster centroids
    pairwise_dist = general_utils.pairwise_distances(centroids, metric)  # (cur_nc, cur_nc)

    # set distance between same cluster to np.inf
    diag_ = np.diag(np.full((pairwise_dist.shape[0],), np.inf))  # (cur_nc, cur_nc)
    pairwise_dist += diag_

    while cur_nc != nc:
        # clusters with min distances
        i, j = np.unravel_index(np.argmin(pairwise_dist), pairwise_dist.shape)  # i != j

        # so combine i-th and j-th clusters
        cur_nc -= 1
        new_label = y[np.argmax(masks[i])]

        # update mask and final labels
        masks[i] += masks[j]
        y[masks[i]] = new_label

        # calculate new centroid
        centroids[i] = np.sum(x[masks[i]], axis=0) / np.sum(masks[i])

        # update pairwise distance
        for k in range(pairwise_dist.shape[0]):
            if k == j or k == i:
                continue

            dist = metric(centroids[i], centroids[k])
            pairwise_dist[i][k] = dist
            pairwise_dist[k][i] = dist

        # delete info about j-th cluster
        masks = np.delete(masks, j, axis=0)
        centroids = np.delete(centroids, j, axis=0)
        pairwise_dist = np.delete(np.delete(pairwise_dist, j, axis=1), j, axis=0)

    labels = np.unique(y)  # (cur_nc,)
    for ind, lab in enumerate(labels):
        y[y == lab] = ind

    return y


def kmeans(x: np.ndarray, nc: int, metric, n_init: int = 10) -> np.ndarray:
    """ Clustering using flat method (kmeans) with distance between centroids

    :param x: shape(n, d), where n - num of examples, d - num of features
    :param nc: number of clusters
    :param metric: callable function for counting metric
    :param n_init: number of time the algorithm will be run with different centroid seeds.
    :return: y: shape(n,), predict labels"""

    max_iter = 1000
    best_inertia = None

    if n_init < 1:
        raise ValueError()

    for ni in range(n_init):
        # random init nc centroids
        centroids = x[np.random.choice(x.shape[0], size=nc, replace=False)]  # (nc,)

        y = np.zeros((x.shape[0],), dtype=np.int32)  # (n,)
        labels = np.arange(nc)[:, np.newaxis]  # (nc, 1)

        for i in range(max_iter):
            prev_centroids = centroids  # (nc,)
            # cluster
            for ind, example in enumerate(x):
                y[ind] = np.argmin([metric(example, centroid) for centroid in centroids])

            # calculate new centroid coordinates
            masks = labels == y  # (nc, n)
            centroids = masks @ x / np.sum(masks, axis=1)[:, np.newaxis]

            if np.all(prev_centroids == centroids):
                break

        # sum of squared distances of example to their closest centroid
        inertia = 0
        for ind, example in enumerate(x):
            inertia += np.min([metric(example, centroid) for centroid in centroids]) ** 2

        if best_inertia is None or inertia < best_inertia:
            best_y = y
            best_inertia = inertia

    return best_y
