import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import matplotlib.pyplot as plt
from embedding.plots import tsne_plot


if __name__ == '__main__':
    '''karate_club = pd.read_csv('/home/varrone/Data/Karate/ucidata-zachary/out.ucidata-zachary', delimiter='\t',
                              header=[0, 1])
    karate_club = np.squeeze(karate_club.values)
    adj = np.zeros((34, 34))
    edges = np.array([list(map(lambda n: int(n) - 1, e.split())) for e in karate_club])
    adj[np.ix_(edges[:, 0], edges[:, 1])] = 1'''
    '''graph = nx.karate_club_graph()

    node2vec = Node2Vec(graph, dimensions=3, walk_length=30, num_walks=200, workers=4)

    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    tsne_plot(model.wv.vectors, filename_save='prova.png')'''
    embs = pd.read_csv('/home/varrone/Repo/node2vec/')


