import argparse
import configparser

import numpy as np

from plots import tsne_plot

config = configparser.ConfigParser()
config.read('/home/varrone/config.ini')

data_folder = config['EMBEDDING']['DATA']

parser = argparse.ArgumentParser()

# ToDo: join plot scripts
# ToDo: add description
parser.add_argument('--embedding-representation', type=str, required=True)
parser.add_argument('--dataset', type=str, default='GSE92743')
parser.add_argument('--save-fig', default=False, action='store_true')

parser.add_argument('-k', '--n-clusters', type=int)
parser.add_argument('--landmarks', type=str, default='l1000')

# t-SNE
parser.add_argument('--perplexity', type=int, default=30)
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

    if args.save_fig:
        filename = 'plots/tsne_' + str(args.embedding_representation) + '.png'

    tsne_plot(embedding, landmarks=landmarks, filename_save=filename, perplexity=args.perplexity)
