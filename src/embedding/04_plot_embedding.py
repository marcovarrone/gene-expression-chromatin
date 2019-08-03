import argparse
import configparser
import warnings

import numpy as np
from MulticoreTSNE import MulticoreTSNE as TSNE
from numba.errors import NumbaDeprecationWarning, NumbaPerformanceWarning, NumbaWarning
from sklearn.decomposition import PCA
from umap import UMAP

from plots import plot

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

config = configparser.ConfigParser()
config.read('/home/varrone/config.ini')

data_folder = config['EMBEDDING']['DATA']

parser = argparse.ArgumentParser()

# ToDo: join plot scripts
# ToDo: add description
parser.add_argument('-e', '--embedding-representation', type=str, required=True)
parser.add_argument('--dataset', type=str, default='GSE92743')
parser.add_argument('--save-fig', default=False, action='store_true')
parser.add_argument('-t', '--technique', choices=['pca', 'tsne', 'umap'])

parser.add_argument('-k', '--n-clusters', type=int)
parser.add_argument('-l', '--landmarks', type=str, default='l1000')

# t-SNE
parser.add_argument('-p', '--perplexity', type=int, default=30)

# UMAP
parser.add_argument('-n', '--n-neighbors', type=int, default=15)

args = parser.parse_args()

if __name__ == '__main__':
    embedding = np.load('embeddings/' + str(args.dataset) + '/' + str(args.embedding_representation) + '.npy')

    if args.n_clusters:
        landmarks = np.load(
            'landmarks/' + str(args.dataset) + '/' + str(args.embedding_representation) + '_k' + str(args.n_clusters) +
            '.npy')
    else:
        landmarks = np.load('landmarks/' + str(args.dataset) + '/' + str(args.landmarks) + '.npy')

    filename = None

    if args.technique == 'pca':
        technique = PCA(n_components=2)
        name = 'pca'
    elif args.technique == 'tsne':
        technique = TSNE(n_jobs=8, verbose=1, n_components=2, random_state=42, perplexity=args.perplexity)
        name = 'tsne_'+str(args.perplexity)
    else:
        technique = UMAP(n_components=2, n_neighbors=args.n_neighbors)
        name = 'umap_'+str(args.n_neighbors)

    if args.save_fig:
        filename = 'plots/' + str(name) + '_' + str(args.embedding_representation) + '.png'

    plot(technique, embedding, landmarks=landmarks, filename_save=filename)
