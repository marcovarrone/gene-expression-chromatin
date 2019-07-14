import argparse
import configparser

import numpy as np

from plots import pca_plot, tsne_plot

config = configparser.ConfigParser()
config.read('/home/varrone/config.ini')

parser = argparse.ArgumentParser()

# ToDo: add description
parser.add_argument('n_samples', type=int)
parser.add_argument('--offset', type=int, nargs='?', default=0)
parser.add_argument('--no-normalize', default=False, action='store_true')
parser.add_argument('--valid-size', type=float, nargs='?', default=0)
parser.add_argument('--technique', default='pca', choices=['pca', 'tsne'])
parser.add_argument('--save-fig', default=False, action='store_true')

args = parser.parse_args()

representation = '_' + str(args.n_samples) + '_' + str(args.offset)
if not args.no_normalize:
    representation += '_normalized'

if args.valid_size > 0:
    representation += '_' + str(args.valid_size)

embedding = np.load('embeddings_from_data/GSE92743' + str(representation) + '.npy')

filename = None


if args.save_fig:
    filename = 'autoencoder_' + str(args.technique) + str(representation)

if args.technique == 'pca':
    pca_plot(embedding, filename_save=filename)
elif args.technique == 'tsne':
    tsne_plot(embedding, filename_save=filename)
