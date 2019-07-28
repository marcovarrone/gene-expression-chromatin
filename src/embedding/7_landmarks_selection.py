import os
import numpy as np

from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans

import argparse

parser = argparse.ArgumentParser()

# ToDo: add description
parser.add_argument('--embedding-representation', type=str, required=True)
parser.add_argument('--dataset', type=str, default='GSE92743')
parser.add_argument('-k', '--n-clusters', type=int, required=True)
parser.add_argument('--save-landmarks', default=False, action='store_true')

args = parser.parse_args()


def clusters_centers(clusters, data):
    centroids = []
    for cluster in clusters:
        coords = data[cluster]
        centroids.append(np.mean(coords, 0))
    return np.array(centroids)


embeddings = np.load('embeddings/' + str(args.dataset) + '/' + str(args.embedding_representation) + '.npy')
if __name__ == '__main__':
    kmeans = KMeans(n_clusters=args.n_clusters, n_jobs=8)
    kmeans.fit_transform(embeddings)

    centers = kmeans.cluster_centers_

    landmarks, _ = pairwise_distances_argmin_min(centers, embeddings)
    if args.save_landmarks:
        np.save('landmarks/' + str(args.dataset) + '/' + str(args.embedding_representation) + '_k' + str(args.n_clusters) + '.npy',
                landmarks)
