import argparse
import os

import networkx as nx

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--embedding-representation', type=str,
                    default='autoencoder_50_e50_lr0.0001_bs128_bn_20000_0_normalized')
parser.add_argument('--dataset', type=str, default='GSE92743')
parser.add_argument('--save-embedding', default=False, action='store_true')
parser.add_argument('--random-seed', type=int, default=42)

parser.add_argument('--threshold', type=float, default=0.0001)

parser.add_argument('--gpu', default=False, action='store_true')
parser.add_argument('--p-test', type=float, default=0.1)
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=15)
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

graphsage = GraphSAGELinkPredictor(graph=g_nx, node_features=node_features, p_test=args.p_test,
                                   learning_rate=args.learning_rate, epochs=args.epochs, save_model=args.save_model,
                                   embedding_representation=args.embedding_representation)
graphsage.fit()

if args.save_embedding:
    embeddings = graphsage.embeddings
    print('Saving embedding in embeddings/' + str(graphsage) + '.npy')
    np.save(
        'embeddings/' + str(args.dataset) + '/' + str(graphsage) + '.npy',
        embeddings, allow_pickle=False)
