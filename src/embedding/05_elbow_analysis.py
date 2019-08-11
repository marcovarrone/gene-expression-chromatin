import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()

parser.add_argument('-e', '--embeddings', type=str, required=True)
parser.add_argument('--dataset', type=str, default='GSE92743')
parser.add_argument('--data-path', type=str)
parser.add_argument('--min-low', type=int, default=10)
parser.add_argument('--max-low', type=int, default=100)
parser.add_argument('--step-low', type=int, default=10)
parser.add_argument('--min-high', type=int, default=100)
parser.add_argument('--max-high', type=int, default=500)
parser.add_argument('--step-high', type=int, default=50)
parser.add_argument('--save-fig', default=False, action='store_true')
parser.add_argument('--save-variance', default=False, action='store_true')

args = parser.parse_args()

def compute_explained_variance(data, k_range):
    print("Computing total sum of squares")
    tss = sum(pdist(data) ** 2) / data.shape[0]
    wcss = [tss]
    for k in k_range:
        print("Computing K-means for k =", k)
        kmean = KMeans(n_clusters=k, n_jobs=10).fit(data)
        centroid = kmean.cluster_centers_
        k_euclid = cdist(data, centroid)
        dist = np.min(k_euclid, axis=1)
        wcss.append(sum(dist ** 2))

    bss = tss - wcss
    explained_variance = bss / tss

    return explained_variance

def plot_elbow(explained_variance, k_range):
    plt.plot(np.hstack(([0], k_range)), explained_variance)

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.ylabel('Explained variance')

    plt.xlabel('N. of clusters')

    plt.show()

if __name__ == '__main__':
    if args.data_path:
        embedding = np.load(args.data_path)
    else:
        embedding = np.load('embeddings/' + str(args.dataset) + '/' + str(args.embeddings) + '.npy')

    k_range_low = np.arange(args.min_low, args.max_low, args.step_low)
    k_range_high = np.arange(args.min_high, args.max_high, args.step_high)
    k_range = np.hstack((k_range_low, k_range_high))

    if os.path.isfile('elbow/' + str(args.dataset) + '/elbow_' + str(args.embeddings) + '_' + str(
            args.min_low) + '_' + str(args.max_high) + '.npy'):
        explained_variance = np.load('elbow/' + str(args.dataset) + '/elbow_' + str(args.embeddings) + '_' + str(
            args.min_low) + '_' + str(args.max_high) + '.npy')
    else:
        explained_variance = compute_explained_variance(embedding, k_range)

    plot_elbow(explained_variance, k_range)

    if args.save_variance:
        np.save('elbow/' + str(args.dataset) + '/elbow_' + str(args.embeddings) + '_' + str(
            args.min_low) + '_' + str(args.max_high) + '.npy',
                explained_variance)

    if args.save_fig:
        plt.savefig('elbow/' + str(args.dataset) + '/elbow_' + str(args.embeddings) + '_' + str(
            args.min_low) + '_' + str(args.max_high) + '.png')

    # kmeans = [KMeans(i) for i in k_range]

    # scores = [kmean.fit(embedding).score(embedding) for kmean in kmeans]

    # print(scores)
