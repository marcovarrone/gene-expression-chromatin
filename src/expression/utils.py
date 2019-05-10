import numpy as np
import scipy.sparse as sps
import tensorflow as tf

DIRECTORY_GRAPH = '/home/varrone/Data/GSE92743_CMap/graphs/'


# DIRECTORY_GRAPH = 'home/nanni/Projects/gexi-top/data/interim/graph/s/'

def get_closest_nodes(adj, n_neighbors):
    nearest_neighbors = np.argpartition(-adj, np.arange(0, n_neighbors))[:, :n_neighbors - 1]
    closest_nodes = np.insert(nearest_neighbors, 0, np.arange(0, nearest_neighbors.shape[0]), axis=1)
    closest_nodes_idx = np.ravel(closest_nodes.flatten())
    return closest_nodes_idx


def get_neighbors_chain(adj, n_neighbors):
    closest_nodes_idx = 943 + get_target_target_neighbors(adj[943:, 943:], n_neighbors)


def get_target_target_neighbors(adj, n_neighbors, indices):
    nearest_neighbors = np.argpartition(-adj, np.arange(0, n_neighbors))[:, :n_neighbors - 1]
    closest_nodes = np.insert(nearest_neighbors, 0, np.arange(0, nearest_neighbors.shape[0]), axis=1)
    closest_nodes_idx = np.ravel(closest_nodes.flatten())
    return closest_nodes_idx


def target_landmark_neighbors(adj, n_neighbors, indices):
    nearest_neighbors = np.argpartition(-adj, np.arange(0, n_neighbors))[:, :n_neighbors - 1]
    closest_nodes = np.insert(nearest_neighbors, 0, indices, axis=1)
    closest_nodes_idx = np.ravel(closest_nodes.flatten())
    return closest_nodes_idx


def get_neighbors(adj, n_neighbors, landmark, target, indices=None):
    if not indices:
        indices = np.arange(0, adj[0])

    adj = np.take(adj, indices, axis=0)

    if target and not landmark:
        adj = adj[943:, 943:]
    elif landmark and not target:
        adj = adj[:943, :943]
    else:
        adj = adj[:, :943]

    nearest_neighbors = np.argpartition(-adj, np.arange(0, n_neighbors))[:, :n_neighbors - 1]
    closest_nodes = np.insert(nearest_neighbors, 0, indices, axis=1)
    closest_nodes_idx = np.ravel(closest_nodes.flatten())
    return closest_nodes_idx


def get_genemania(landmarks=True, targets=False):
    graph = sps.load_npz(str(DIRECTORY_GRAPH) + 'GSE92743_CMap_genemania.npz')
    graph = graph.todense()
    graph = graph + graph.T - np.diag(graph.diagonal())
    if landmarks and not targets:
        return graph[:943, :943]
    elif targets and not landmarks:
        return graph[943:, 943:]
    else:
        return graph


def join_source_genes(X, y, n_targets, idxs=None):
    if idxs is None:
        idxs = np.random.choice(range(y.shape[1]), n_targets)
    y_targets = y[:, idxs]
    X_targets = np.hstack((X, np.delete(y, idxs, axis=1)))
    return X_targets, y_targets


def get_H(R, threshold):
    R[R < threshold] = 0
    R = sps.coo_matrix(R)

    n_edges = R.nnz
    i = np.arange(n_edges)

    data_order = R.data.argsort()
    row = R.row[data_order]
    col = R.col[data_order]
    data = R.data[data_order]

    H = np.zeros((n_edges, R.shape[0]), dtype=np.float32)
    H[i, row] = np.abs(data)
    H[i, col] = - np.sign(data) * np.abs(data)
    return H


def to_sparse_tensor(A):
    A = sps.coo_matrix(A)
    indices = np.concatenate((np.expand_dims(A.row, 1),
                              np.expand_dims(A.col, 1)), 1)
    return tf.SparseTensor(indices=indices, values=A.data, dense_shape=A.shape)


def delete_csr(mat, rows, axis):
    if not isinstance(mat, sps.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")

    if rows is None:
        return mat

    mask = np.ones(mat.shape[axis], dtype=bool)
    mask[rows] = False
    return mat[mask]

