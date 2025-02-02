import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def generate_data(n_samples=20, n_features=2, n_clusters=3, random_state=42):
    data, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state)
    return data

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def agglomerative_clustering(data):
    n_samples = data.shape[0]
    clusters = {i: [i] for i in range(n_samples)}  # Изначально каждый элемент — отдельный кластер
    distances = np.full((n_samples, n_samples), np.inf)

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            distances[i, j] = distances[j, i] = euclidean_distance(data[i], data[j])

    linkage_matrix = []

    while len(clusters) > 1:
        min_dist = np.inf
        pair_to_merge = None

        for i in clusters:
            for j in clusters:
                if i < j:
                    avg_dist = np.mean([distances[p1, p2] for p1 in clusters[i] for p2 in clusters[j]])
                    if avg_dist < min_dist:
                        min_dist = avg_dist
                        pair_to_merge = (i, j)

        c1, c2 = pair_to_merge
        new_cluster = clusters[c1] + clusters[c2]
        new_index = max(clusters.keys()) + 1
        clusters[new_index] = new_cluster

        del clusters[c1]
        del clusters[c2]

        linkage_matrix.append([c1, c2, min_dist, len(new_cluster)])

    return np.array(linkage_matrix)


def plot_dendrogram(matrix):
    plt.figure(figsize=(10, 7))
    from scipy.cluster.hierarchy import dendrogram
    dendrogram(matrix, orientation='top', distance_sort='ascending', show_leaf_counts=True)
    plt.title('Dendrogram')
    plt.xlabel('Виды')
    plt.ylabel('Distance')
    plt.show()


# 4. Основной код
if __name__ == "__main__":
    data = generate_data(n_samples=30, n_features=2, n_clusters=3, random_state=42)

    linkage_matrix = agglomerative_clustering(data)

    plot_dendrogram(linkage_matrix)
