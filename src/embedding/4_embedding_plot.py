import argparse
import configparser

import numpy as np

from plots import pca_plot, tsne_plot

config = configparser.ConfigParser()
config.read('/home/varrone/config.ini')

data_folder = config['EMBEDDING']['DATA']

parser = argparse.ArgumentParser()

# ToDo: add description
parser.add_argument('embedding-representation')
parser.add_argument('--dataset', type=str, default='GSE92743')

args = parser.parse_args()

if __name__ == '__main__':
    embedding = np.load('embeddings/' + str(args.dataset) + '/' + str(args.embedding_representation) + '.npy')

    filename = None

    if args.save_fig:
        filename = 'plots/' + str(args.technique) + '_' + str(args.data_representation) + '_'+ str(
            args.embedding_representation)

    if args.technique == 'pca':
        pca_plot(embedding, filename_save=filename)
    elif args.technique == 'tsne':
        tsne_plot(embedding, filename_save=filename)
