import argparse
import os

import networkx as nx
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-data-repr', '--data-representation', type=str, default='20000_0_normalized')
parser.add_argument('-emb-repr', '--embedding-representation', type=str, default='autoencoder_50_e120_lr0.0001_bs128_bn')
parser.add_argument('--dataset', type=str, default='GSE92743')
parser.add_argument('--save-embedding', default=False, action='store_true')

parser.add_argument('--gpu', default=False, action='store_true')

parser.add_argument('--threshold', type=float, default=0.0001)

args = parser.parse_args()

if not args.gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import configparser
import stellargraph as sg
import scipy.sparse as sps
import numpy as np
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator, GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE, link_classification

import keras

config = configparser.ConfigParser()
config.read('/home/varrone/config.ini')

data_folder = config['EMBEDDING']['DATA']

adjacency = sps.load_npz(config['GRAPH']['GENEMANIA_NPZ']).todense()
adjacency[adjacency < args.threshold] = 0

g_nx = nx.from_numpy_matrix(adjacency)

X_train = np.load(str(data_folder) + '/' + str(args.dataset) + '/' + str(args.data_representation) + '.npy')

node_features = np.load('embeddings/' + str(args.dataset) + '/' + str(args.embedding_representation) + '.npy')

for n in g_nx:
    g_nx.nodes[n]['feature'] = node_features[n]

G = sg.StellarGraph(g_nx, node_features='feature')

# Define an edge splitter on the original graph G:
edge_splitter_test = EdgeSplitter(g_nx)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G, and obtain the
# reduced graph G_test with the sampled links removed:
G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
    p=0.1, method="global", keep_connected=True
)

# Define an edge splitter on the reduced graph G_test:
edge_splitter_train = EdgeSplitter(G_test)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G_test, and obtain the
# reduced graph G_train with the sampled links removed:
G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
    p=0.1, method="global", keep_connected=True
)

G_train = sg.StellarGraph(G_train, node_features="feature")
G_test = sg.StellarGraph(G_test, node_features="feature")

batch_size = 10
epochs = 20
num_samples = [100, 10]
edge_embedding_method = 'hadamard'

train_gen = GraphSAGELinkGenerator(G_train, batch_size, num_samples).flow(
    edge_ids_train, edge_labels_train, shuffle=True
)

test_gen = GraphSAGELinkGenerator(G_test, batch_size, num_samples).flow(
    edge_ids_test, edge_labels_test
)

layer_sizes = [50, 50]
assert len(layer_sizes) == len(num_samples)

graphsage = GraphSAGE(
    layer_sizes=layer_sizes, generator=train_gen, bias=True, dropout=0.3
)

x_inp, x_out = graphsage.build()

prediction = link_classification(
    output_dim=1, output_act="relu", edge_embedding_method='hadamard'
)(x_out)

model = keras.Model(inputs=x_inp, outputs=prediction)

model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-4),
    loss=keras.losses.binary_crossentropy,
    metrics=["acc"],
)

history = model.fit_generator(
    train_gen,
    epochs=epochs,
    validation_data=test_gen,
    verbose=1,
    use_multiprocessing=True,
    workers=10,
)

if args.save_embedding:
    x_inp_src = x_inp[0::2]
    x_out_src = x_out[0]

    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
    node_features = pd.DataFrame(node_features)
    node_ids = node_features.index
    node_gen = GraphSAGENodeGenerator(G, batch_size, num_samples).flow(node_ids)

    emb = embedding_model.predict_generator(node_gen, workers=4, verbose=1)
    node_embeddings = emb[:, 0, :]
    print('Saving embedding in embeddings/' + str(args.data_representation) + '_' + str(
        args.embedding_representation) + '_graphsage.npy')
    np.save('embeddings/' + str(args.data_representation) + '_' + str(args.embedding_representation) + '_graphsage.npy',
            node_embeddings)
