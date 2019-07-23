import argparse
import os

import networkx as nx
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-data-repr', '--data-representation', type=str, default='20000_0_normalized')
parser.add_argument('-emb-repr', '--embedding-representation', type=str, default='autoencoder_50_e120_lr0.0001_bs128_bn')
parser.add_argument('--dataset', type=str, default='GSE92743')
parser.add_argument('--save-embedding', default=False, action='store_true')
parser.add_argument('--random-seed', type=int, default=42)

parser.add_argument('--threshold', type=float, default=0.0001)

parser.add_argument('--gpu', default=False, action='store_true')
parser.add_argument('--save-model', default=False, action='store_true')


args = parser.parse_args()

if not args.gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import configparser
import stellargraph as sg
import scipy.sparse as sps
import numpy as np
import tensorflow as tf
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator, GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE, link_classification
from graphsage import GraphSAGELinkPredictor

import keras

config = configparser.ConfigParser()
config.read('/home/varrone/config.ini')

np.random.seed(args.random_seed)
tf.set_random_seed(args.random_seed)

adjacency = sps.load_npz(config['GRAPH']['GENEMANIA_NPZ']).todense()
adjacency[adjacency < args.threshold] = 0

g_nx = nx.from_numpy_matrix(adjacency)

node_features = np.load('embeddings/' + str(args.dataset) + '/' + str(args.embedding_representation) + '.npy')

graphsage = GraphSAGELinkPredictor(graph=g_nx, node_features=node_features, p_test=0.0, learning_rate=1e-5)
graphsage.fit()

if args.save_embedding:
    embeddings = graphsage.embeddings
    print('Saving embedding in embeddings/' + str(args.data_representation) + '_' + str(
        args.embedding_representation) + '_graphsage.npy')
    np.save('embeddings/' + str(args.data_representation) + '_' + str(args.embedding_representation) + '_graphsage.npy',
            embeddings)
