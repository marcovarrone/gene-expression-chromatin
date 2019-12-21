import os
import argparse
import networkx as nx
import numpy as np
import sherpa
import stellargraph as sg
import wandb
from keras import optimizers, losses, metrics, Model
from stellargraph import globalvar
from stellargraph.data import EdgeSplitter
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.mapper import GraphSAGELinkGenerator
from wandb.keras import WandbCallback
from models.graphsage_link_prediction import train, test
from embedding.plots import tsne_plot

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.environ["OMP_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10"
os.environ["NUMEXPR_MAX_THREADS"] = "10"


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #parser.add_argument('--hidden-sizes', nargs='*', type=int)
    parser.add_argument('--hidden-size-1', type=int)
    parser.add_argument('--hidden-size-2', type=int)
    #parser.add_argument('--n-samples', nargs='*', type=int)
    parser.add_argument('--n-samples-1', type=int)
    parser.add_argument('--n-samples-2', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--num-walks', type=int)
    parser.add_argument('--walk-length', type=int)

    args = parser.parse_args()

    adj_hic = np.load(
        '/home/varrone/Prj/gene-expression-chromatin/src/link_prediction/data/GM19238/interactions/interactions_chr_02_90.npy')
    graph_hic = nx.from_numpy_array(adj_hic)
    n_nodes = nx.number_of_nodes(graph_hic)

    degrees = np.array(list(dict(graph_hic.degree()).values()))

    betweenness = np.array(list(nx.betweenness_centrality(graph_hic, normalized=True).values()))

    clustering = np.array(list(nx.clustering(graph_hic).values()))

    node_features = np.vstack((degrees, betweenness, clustering)).T
    node_ids = np.arange(node_features.shape[0])
    for nid, f in zip(node_ids, node_features):
        graph_hic.node[nid][globalvar.FEATURE_ATTR_NAME] = f
    wandb.init(project='graphsage-tuning')
    model, node_embeddings = train(
        graph_hic,
        [args.hidden_size_1, args.hidden_size_2],
        [args.n_samples_1, args.n_samples_2],
        args.batch_size,
        args.epochs,
        args.learning_rate,
        args.dropout,
        #number_of_walks=args.num_walks,
        #walk_length=args.walk_length,
        wandb=wandb
    )
    '''sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    parameters = [sherpa.Ordinal(name='layer_size1', range=sizes),
                  sherpa.Ordinal(name='layer_size2', range=sizes),
                  sherpa.Discrete(name='num_samples1', range=[5, 100]),
                  sherpa.Discrete(name='num_samples2', range=[5, 100]),
                  sherpa.Ordinal(name='batch_size', range=[16, 32, 64]),
                  sherpa.Continuous(name='learning_rate', range=[1e-4, 1e-2], scale='log'),
                  sherpa.Continuous(name='dropout', range=[0.0, 0.5])
                  ]

    algorithm = sherpa.algorithms.GPyOpt(max_num_trials=150,num_initial_data_points=15)

    study = sherpa.Study(parameters=parameters,
                         algorithm=algorithm,
                         lower_is_better=False)

    for trial in study:
        run = wandb.init(project='graphsage-tuning', reinit=True)
        p = trial.parameters
        params = {
            'layer_size': [int(p['layer_size1']), int(p['layer_size2'])],
            'num_samples': [int(p['num_samples1']), int(p['num_samples2'])],
            'batch_size': p['batch_size'],
            'num_epochs': 40,
            'learning_rate': p['learning_rate'],
            'dropout': p['dropout']
        }
        run.config.update(p)
        model, _ = train(graph_hic, **params, study=study, trial=trial, wandb=wandb)
        run.state = "finished"

        test_accuracy = test(graph_hic, model, batch_size=p['batch_size'])
        study.add_observation(trial=trial,
                              iteration=0,
                              objective=test_accuracy)
        study.finalize(trial)'''

    #node_embeddings = node_embeddings.reshape((node_embeddings.shape[0], node_embeddings.shape[2]))
    tsne_plot(node_embeddings, landmarks=np.arange(node_embeddings.shape[0]), gradient=True, filename_save='wandb_tsne.png')
    wandb.log({'emb_tsne': wandb.Image('wandb_tsne.png', caption='Embedding t-SNE')})
    # np.save('embeddings_graphsage.npy', node_embeddings)

    wandb.log({'test_accuracy': test(graph_hic, model, 50)})
