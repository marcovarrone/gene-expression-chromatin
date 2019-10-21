from typing import AnyStr, List
import os
import argparse

import networkx as nx
import numpy as np
from stellargraph import globalvar
from embedding.plots import tsne_plot, pca_plot
import tensorflow as tf
from models.graphsage_link_prediction import train, test
import matplotlib.pyplot as plt



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.environ["OMP_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10"



if __name__ == '__main__':
    tf.set_random_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--layer-size', nargs='*', type=int)
    parser.add_argument('--num-samples', nargs='*', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--dropout', type=float)

    args = parser.parse_args()

    adj_hic = np.load(
        '/home/varrone/Prj/gene-expression-chromatin/src/link_prediction/data/GM19238/interactions_90.npy')
    graph_hic = nx.from_numpy_array(adj_hic)

    n_nodes = nx.number_of_nodes(graph_hic)

    degrees = np.array(list(dict(graph_hic.degree()).values()))

    betweenness = np.array(list(nx.betweenness_centrality(graph_hic, normalized=True).values()))

    clustering = np.array(list(nx.clustering(graph_hic).values()))

    #node_features = np.vstack((degrees, betweenness, clustering)).T
    node_features = degrees.reshape(len(degrees), 1)
    node_ids = np.arange(node_features.shape[0])
    for nid, f in zip(node_ids, node_features):
        graph_hic.node[nid][globalvar.FEATURE_ATTR_NAME] = f

    model, node_embeddings = train(
        graph_hic,
        args.layer_size,
        args.num_samples,
        args.batch_size,
        args.epochs,
        args.learning_rate,
        args.dropout,
    )

    node_embeddings = node_embeddings.reshape((node_embeddings.shape[0], node_embeddings.shape[2]))
    save_str = "_n{}_l{}_d{}_r{}".format(
        "_".join([str(x) for x in args.num_samples]),
        "_".join([str(x) for x in args.layer_size]),
        args.dropout,
        args.learning_rate)
    #np.save('embeddings_graphsage_slim'+save_str, node_embeddings)
    pca_plot(node_embeddings)
    tsne_plot(node_embeddings)
    coexpression = np.load(
        '/home/varrone/Prj/gene-expression-chromatin/src/link_prediction/data/GM19238/coexpression_90.npy')
    graph_coexp = nx.from_numpy_array(coexpression)

    print(test(graph_hic, model, 64))
