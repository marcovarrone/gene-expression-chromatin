import argparse
import os
import warnings

import networkx as nx
import numpy as np
from stellargraph import globalvar

from embedding.plots import tsne_plot
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
from models.graphsage_link_prediction import train, test

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
    parser.add_argument('--dataset', type=str, default='GM19238')
    parser.add_argument('--interactions', type=str, required=True)
    parser.add_argument('--threshold', type=int, default=90)
    parser.add_argument('--layer-size', nargs='*', type=int)
    parser.add_argument('--num-samples', nargs='*', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--val-size', type=float, default=0.1)
    parser.add_argument('--train-size', type=float, default=0.1)
    #parser.add_argument('--num-walks', type=int)
    #parser.add_argument('--walk-length', type=int)
    parser.add_argument('--norm', default=False, action='store_true')
    parser.add_argument('--featureless', default=False, action='store_true')
    parser.add_argument('--wandb', default=None, action='store_true')

    args = parser.parse_args()

    interactions_path = './data/{}/interactions/interactions_{}.npy'.format(
        args.dataset, args.interactions)

    graph_hic = nx.from_numpy_array(np.load(interactions_path))

    n_nodes = nx.number_of_nodes(graph_hic)

    if args.featureless:
        node_features = np.eye(n_nodes)
    else:
        degrees = np.array(list(dict(graph_hic.degree()).values()))
        betweenness = np.array(list(nx.betweenness_centrality(graph_hic, normalized=True).values()))
        clustering = np.array(list(nx.clustering(graph_hic).values()))

        node_features = np.vstack((degrees, betweenness, clustering)).T
    node_ids = np.arange(node_features.shape[0])
    for nid, f in zip(node_ids, node_features):
        graph_hic.node[nid][globalvar.FEATURE_ATTR_NAME] = f

    #if args.wandb:
        #wandb.init(project='graphsage-tuning')

    model, node_embeddings = train(
        graph_hic,
        args.layer_size,
        args.num_samples,
        args.batch_size,
        args.epochs,
        args.learning_rate,
        args.dropout,
        #wandb=wandb
        #args.num_walks,
        #args.walk_length
    )

    # node_embeddings = node_embeddings.reshape((node_embeddings.shape[0], node_embeddings.shape[2]))
    save_str = "{}_n{}_l{}_d{}_r{}_d{}_d{}{}{}".format(
        args.interactions,
        "_".join([str(x) for x in args.num_samples]),
        "_".join([str(x) for x in args.layer_size]),
        args.dropout,
        args.learning_rate,
        args.num_walks if args.num_walks else 0,
        args.walk_length if args.walk_length else 0,
        '_norm' if args.norm else '',
        '_featureless' if args.featureless else '')
    np.save('embeddings/graphsage/{}'.format(save_str), node_embeddings)
    # pca_plot(node_embeddings)
    tsne_plot(node_embeddings, landmarks=np.arange(node_embeddings.shape[0]), gradient=True)
    # coexpression = np.load(
    #   '/home/varrone/Prj/gene-expression-chromatin/src/link_prediction/data/GM19238/coexpression_chr4_90.npy')
    # graph_coexp = nx.from_numpy_array(coexpression)

    print(test(graph_hic, model, 64))
