import math

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def main():
    flowers = load_iris()
    data = flowers.data
    targets = flowers.target

    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(data)

    neigh = KNeighborsClassifier(n_neighbors=math.trunc(np.sqrt(len(data))))
    neigh.fit(data, targets)
    point = [4, 3, 1, 5]
    predicted_target = neigh.predict([point])
    pca_new = pca.transform([point])

    transformed_data = np.concatenate([transformed_data, pca_new])
    targets = np.concatenate([targets, predicted_target])

    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=targets)
    plt.show()

if __name__ == '__main__':
    main()