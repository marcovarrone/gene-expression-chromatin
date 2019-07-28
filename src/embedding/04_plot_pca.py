import argparse
import configparser

import numpy as np

from plots import pca_plot

config = configparser.ConfigParser()
config.read('/home/varrone/config.ini')

data_folder = config['EMBEDDING']['DATA']

parser = argparse.ArgumentParser()

# ToDo: add description
parser.add_argument('--embedding-representation', type=str, required=True)
parser.add_argument('--dataset', type=str, default='GSE92743')
parser.add_argument('--save-fig', default=False, action='store_true')

args = parser.parse_args()

if __name__ == '__main__':
    embedding = np.load('embeddings/' + str(args.dataset) + '/' + str(args.embedding_representation) + '.npy')

    filename = None

    if args.save_fig:
        filename = 'plots/pca_' + str(args.embedding_representation) + '.png'

    pca_plot(embedding, filename_save=filename)
