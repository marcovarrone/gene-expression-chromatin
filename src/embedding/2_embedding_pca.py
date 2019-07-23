import argparse
import configparser

import numpy as np
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()

# ToDo: add description and parameters for training
parser.add_argument('-data-repr', '--data-representation', type=str, default='')
parser.add_argument('--dataset', type=str, default='GSE92743')
parser.add_argument('--save-embedding', default=False, action='store_true')
parser.add_argument('--random-seed', type=int, default=42)
parser.add_argument('--save-model', default=False, action='store_true')

parser.add_argument('--explained-variance', type=float, default=0.9)

args = parser.parse_args()

config = configparser.ConfigParser()
config.read('/home/varrone/config.ini')

np.random.seed(args.random_seed)

data_folder = config['EMBEDDING']['DATA']

X_train = np.load(
    str(data_folder) + '/' + str(args.dataset) + '/X_train_genes_' + str(args.data_representation) + '.npy')

pca = PCA()
embedding = pca.fit_transform(X_train)

if 1 > args.explained_variance > 0:
    variance_cum = np.cumsum(pca.explained_variance_ratio_)
    print(pca.explained_variance_ratio_)
    last_index = np.argwhere(variance_cum > args.explained_variance)[0][0]
    print("Take", last_index, "components which explain", str(args.explained_variance*100)+'% of the variance')
    embedding = embedding[:, :last_index]

pca_repr = 'pca_' + str(args.explained_variance)

if args.save_embedding:
    print('Saving embedding in embeddings/' + str(args.dataset) + '/' + str(pca_repr) + '.npy')
    np.save('embeddings/' + str(args.dataset) + '/' + str(pca_repr) + '.npy', embedding)
