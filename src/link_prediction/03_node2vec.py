import argparse
import numpy as np
import networkx as nx
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder
import pickle
import matplotlib.pyplot as plt
from embedding.plots import tsne_plot, pca_plot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='GM19238')
    parser.add_argument('--interactions', type=str, required=True)
    parser.add_argument('--dimensions', type=int, default=8)
    parser.add_argument('--walk-length', type=int)
    parser.add_argument('--p', type=float)
    parser.add_argument('--q', type=float)
    parser.add_argument('--num-walks', type=int)
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()

    interactions_path = './data/{}/interactions/interactions_{}.npy'.format(
        args.dataset, args.interactions)

    graph = nx.from_numpy_array(np.load(interactions_path))

    embeddings = []

    for epoch in range(10):
        node2vec = Node2Vec(
            graph,
            dimensions=args.dimensions,
            walk_length=args.walk_length,
            workers=args.workers,
            p=args.p,
            q=args.q,
            num_walks=args.num_walks)
        model = node2vec.fit(window=10, min_count=1)
        print(model['2'])

    #plt.scatter(model.wv.vectors[:, 0], model.wv.vectors[:, 1])
    #plt.savefig('prova.png')
    #tsne_plot(model.wv.vectors, filename_save='prova.png')
    #pca_plot(model.wv.vectors, filename_save='prova.png')

    #np.save('embeddings_node2vec.npy', model.wv.vectors)


    edges_embs = HadamardEmbedder(keyed_vectors=model.wv)

    with open('/home/varrone/Prj/gene-expression-chromatin/src/link_prediction/data/GM19238/edge_emb_n2v.pkl', 'wb') as file_save:
        pickle.dump(edges_embs, file_save)

    #edges_kv = edges_embs.as_keyed_vectors()

    #edges_kv.save_word2vec_format('/home/varrone/Prj/gene-expression-chromatin/src/link_prediction/data/GM19238/edge_emb_n2v')
