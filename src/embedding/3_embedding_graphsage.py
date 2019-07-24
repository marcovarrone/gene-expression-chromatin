import argparse
import os

import networkx as nx

parser = argparse.ArgumentParser()
parser.add_argument('-data-repr', '--data-representation', type=str, default='20000_0_normalized')
parser.add_argument('-emb-repr', '--embedding-representation', type=str,
                    default='autoencoder_50_e50_lr0.0001_bs128_bn_20000_0_normalized')
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
import scipy.sparse as sps
import numpy as np
import tensorflow as tf
from graphsage import GraphSAGELinkPredictor

config_tf = tf.compat.v1.ConfigProto(device_count={'GPU': 0, 'CPU': 10})
config_tf.intra_op_parallelism_threads = 10
config_tf.inter_op_parallelism_threads = 10
tf.compat.v1.Session(config=config_tf)

config = configparser.ConfigParser()
config.read('/home/varrone/config.ini')

np.random.seed(args.random_seed)
tf.set_random_seed(args.random_seed)

adjacency = sps.load_npz(config['GRAPH']['GENEMANIA_NPZ']).todense()
adjacency[adjacency < args.threshold] = 0
adjacency[adjacency >= args.threshold] = 1

g_nx = nx.from_numpy_matrix(adjacency)

node_features = np.load('embeddings/' + str(args.dataset) + '/' + str(args.embedding_representation) + '.npy')

# ToDo: parameters from command line
graphsage = GraphSAGELinkPredictor(graph=g_nx, node_features=node_features, p_test=0.1, learning_rate=1e-4, epochs=12,
                                   save_model=args.save_model)
graphsage.fit()

# ToDo: fix embedding name saving

if args.save_embedding:
    embeddings = graphsage.embeddings
    print('Saving embedding in embeddings/' + str(args.data_representation) + '_' + str(
        args.embedding_representation) + '_graphsage.npy')
    np.save('embeddings/' + str(args.data_representation) + '_' + str(args.embedding_representation) + '_graphsage.npy',
            embeddings)