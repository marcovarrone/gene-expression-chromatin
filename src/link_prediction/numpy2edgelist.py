import numpy as np
import networkx as nx

if __name__ == '__main__':
    #for chrom in range(1, 10):
    chrom= 2
    interactions = np.load(
        '/home/varrone/Prj/gene-expression-chromatin/src/link_prediction/data/GM19238/interactions/interactions_chr_{:02d}_70.npy'.format(
            chrom))
    Gnx = nx.from_numpy_matrix(interactions)
    nx.write_edgelist(Gnx,'/home/varrone/Prj/gene-expression-chromatin/src/link_prediction/data/GM19238/interactions/interactions_chr_{:02d}_70.edgelist'.format(chrom))
