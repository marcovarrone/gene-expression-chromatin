import networkx as nx
import matplotlib.pyplot as plt

from chiapet_to_sparse import ChiaPetInteractions


def build_graph(A):
    return nx.from_numpy_matrix(A)


def main():
    bin_length = 10000000
    file_path = 'ENCSR000BZX_HCT116_POLR2A.bed'

    contact_matrix = ChiaPetInteractions(file_path, bin_length, different_chrs=False)
    heatmap = contact_matrix.generate_heatmap()
    heatmap = contact_matrix.remove_top_outliers(heatmap=heatmap, ratio=0.005)
    G = build_graph(heatmap)
    G.remove_nodes_from(list(nx.isolates(G)))
    pos = nx.spring_layout(G, k=0.5, iterations=20)
    plt.figure(figsize=(20,20))

    nx.draw(G, pos, node_size=300, with_labels=False)

if __name__ == '__main__':
    main()
