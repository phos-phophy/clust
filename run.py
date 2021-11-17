import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import clustering, metrics, scores
from sklearn.metrics import calinski_harabasz_score, adjusted_rand_score, silhouette_score


def get_param():
    clust = int(input('Select clustering method:\n1) KMeans\n2) Agglomerative\n1 or 2:  ').strip())
    if clust != 1 and clust != 2:
        raise ValueError()

    score = int(input('\nSelect inner score:\n1) Silhouette Score\n2) Calinski-Harabasz Index\n1 or 2:  ').strip())
    if score != 1 and score != 2:
        raise ValueError()

    metric = int(input('\nSelect metric:\n1) l1 (Manhattan)\n2) l2 (Euclidean)\n1 or 2:  ').strip())
    if metric != 1 and metric != 2:
        raise ValueError()

    name = ('KMeans, ' if clust == 1 else 'Agglomerative, ') + \
           ('silhouette, ' if score == 1 else 'calinski-harabasz, ') + ('l1' if metric == 1 else 'l2')

    clust = clustering.kmeans if clust == 1 else clustering.agglomerative
    metric = metrics.l1 if metric == 1 else metrics.l2

    return clust, score, metric, name


def main():
    with open('data/s2.txt', encoding='utf-8') as file:
        data = np.array([list(map(float, line.split())) for line in file])
    with open('data/s2-label.pa', encoding='utf-8') as file:
        gold_labels = np.array([int(line.strip()) for line in file])

    remove_labels = [1, 3, 8, 9, 11, 13, 14]
    for rl in remove_labels:
        mask = gold_labels != rl
        data = data[mask]
        gold_labels = gold_labels[mask]

    print(f'All examples = {gold_labels.shape[0]}\n')

    clust, score, metric, name = get_param()

    # clustering: nc - number of clusters
    best_sc = None
    num_c = list(range(4, 10))
    with tqdm(total=len(num_c)) as bar:
        for nc in num_c:
            predict = clust(data, nc, metrics.l1)

            # count inner scores
            if score == 1:
                sc = scores.silhouette_score(data, predict, metric)
                sklearn_sc = silhouette_score(data, predict, metric=metric)
            else:
                sc = scores.calinski_harabasz_score(data, predict)
                sklearn_sc = calinski_harabasz_score(data, predict)

            if best_sc is None or sc > best_sc:
                best_sc = sc
                best_predict = predict

            bar.set_postfix({'best_score': best_sc, 'score': sc, 'sklearn score': sklearn_sc})
            bar.update()

    # count external scores
    ri = scores.rand_score(gold_labels, best_predict)
    ari = scores.adjusted_rand_score(gold_labels, best_predict)
    ji = scores.jaccard_score(gold_labels, best_predict)

    print(f'\nRI = {ri}\nARI = {ari}\nJI = {ji}')
    print(f'ARI from sklearn = {adjusted_rand_score(gold_labels, best_predict)}')

    plt.figure(figsize=(15, 15))

    labels = np.unique(best_predict)[:, np.newaxis]  # (nc, 1)
    masks = labels == best_predict  # (nc, n)
    centroids = masks @ data / np.sum(masks, axis=1)[:, np.newaxis]

    sns.scatterplot(data[:, 0], data[:, 1], hue=best_predict, legend='full', style=best_predict, palette='bright')
    for ind, cc in enumerate(centroids):
        plt.text(cc[0], cc[1], ind, fontsize=14, fontweight='semibold')
    plt.xlabel(f'RI = {ri}, ARI = {ari}, JI = {ji}', fontsize=11)
    plt.title(name)
    plt.show()


if __name__ == '__main__':
    main()
