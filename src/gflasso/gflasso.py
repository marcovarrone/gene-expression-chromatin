import numpy as np
import scipy.sparse as sps
from sklearn.linear_model import MultiTaskLasso
from data_generator import fake_dataset
from sklearn.metrics import mean_absolute_error

L1_REG = 0.01
GAMMA = 0.01
N_SAMPLES = 20000
N_LANDMARKS = 500
N_TARGETS = 1000

X_train = np.load('/home/nanni/Projects/gexi-top/data/processed/d-gex/bgedv2_X_tr_float64.npy')
y_train = np.load('/home/nanni/Projects/gexi-top/data/processed/d-gex/bgedv2_Y_tr_float64.npy')

X_val = np.load('/home/nanni/Projects/gexi-top/data/processed/d-gex/bgedv2_X_va_float64.npy')
y_val = np.load('/home/nanni/Projects/gexi-top/data/processed/d-gex/bgedv2_Y_va_float64.npy')

X_train_small = X_train[:N_SAMPLES, :N_LANDMARKS]
y_train_small = y_train[:N_SAMPLES, :N_TARGETS]

X_val_small = X_val[:, :N_LANDMARKS]
y_val_small = y_val[:, :N_TARGETS]


def get_H(R):
    # Get upper triangular matrix (diagonal excluded)
    _R = R * (np.ones(R.shape) - np.tri(*R.shape))

    _R = sps.coo_matrix(_R)

    n_edges = _R.nnz
    i = np.arange(n_edges)

    # Construct matrix (n_edges, n_nodes)
    H = np.zeros((n_edges, R.shape[0]))

    # Fill H elements according to the paper
    data_order = _R.data.argsort()
    row = _R.row[data_order]
    col = _R.col[data_order]
    data = _R.data[data_order]
    H[i, row] = np.abs(data)
    H[i, col] = - np.sign(data) * np.abs(data)
    return H


def get_Lu(X, C, l1_reg, gamma, mu):
    lambda_max = np.linalg.eigvals(X.T.dot(X))[0]

    # Implementation from the R library
    # return lambda_max + (1 / mu) * (l1_reg ** 2 + 2 * gamma ** 2 * np.max(C.sum(axis=1)))

    # Implementation from the paper
    return lambda_max + (np.linalg.norm(C) ** 2) / mu


def get_optimal_alpha(C, W, mu):
    alpha = C.dot(W.T) / mu
    alpha[alpha > 1] = 1
    alpha[alpha < -1] = -1
    return alpha.T


def get_grad_f(X, Y, W, C, mu):
    alpha = get_optimal_alpha(C, W, mu)
    return X.T.dot(X.dot(W) - Y) + alpha.dot(C)


# ToDo: understand
def soft_threshold(v, amount):
    result = np.copy(v)
    result[v > amount] = v[v > amount] - amount
    result[v < -amount] = v[v < -amount] + amount
    result[(v > -amount) & (v < amount)] = 0
    return result


def get_B_next(W, grad_f, Lu, l1_reg):
    B = W - (1 / Lu) * grad_f
    return soft_threshold(B, l1_reg / Lu)


def objective(X, B, Y, C, l1_reg) -> int:
    error = np.sum((Y - X.dot(B)) ** 2) / 2
    reg_1 = l1_reg * np.sum(np.abs(B))
    reg_2 = np.sum(np.abs(B.dot(C.T)))
    return error + reg_1 + reg_2


def compute_gradient(X, Y, C, iter_max, mu, Lu, l1_reg, delta_conv, verbose=0, B=None):
    J = X.shape[1]  # input features
    K = Y.shape[1]  # output features

    # ToDo: better way?
    iter_max = int(iter_max)

    if B is None:
        B = np.zeros((J, K))
    W = np.copy(B)

    obj = np.zeros(iter_max)
    # ToDo: something more elegant
    mae = 100
    theta = 1
    for iter in range(iter_max):
        grad_f = get_grad_f(X, Y, W, C, mu)
        B_next = get_B_next(W, grad_f, Lu, l1_reg)
        theta_next = 2 / (iter + 2)
        delta_temp = ((1 - theta) / theta) * theta_next * (B_next - B)
        W = B_next + delta_temp
        delta = np.sum(np.abs(B_next - B))
        B = B_next
        obj[iter] = objective(X, B, Y, C, l1_reg)
        if iter % 100 == 0:
            print('Iteration', iter, ": ", obj[iter])

            mae_new = mean_absolute_error(y_val_small, X_val_small.dot(B))
            if mae_new > mae:
                break

            mae = mae_new
            print('Validation error:', mae)
        # ToDo: is the second condition useful?
        if (delta < delta_conv) or (iter > iter_max):
            break
        theta = theta_next
    return B, obj


def gflasso_2010(X, Y, R, l1_reg=1.0, delta_conv=1e-2, eps=0.0005, gamma=1.0, iter_max=1e4, verbose=False):
    J = X.shape[1]  # input features
    K = Y.shape[1]  # output features

    # Get le penalty matrix
    C = gamma * get_H(R)

    E = C.shape[0]  # Number of edges

    # |E|/2 according to 2012 paper, 1/2 J(K + |E|) for the 2010 paper
    # D = J*(K + E) / 2
    # ToDo: why C.shape[1] instead of C.shape[0], otherwise is J instead of |E|
    D = (1 / 2) * X.shape[1] * (Y.shape[1] + C.shape[1] / 2)
    mu = eps / (2 * D)

    Lu = get_Lu(X, C, l1_reg, gamma, mu)

    B, obj = compute_gradient(X, Y, C, iter_max, mu, Lu, l1_reg, delta_conv)
    return B, obj


if __name__ == '__main__':
    R = np.load('../data/correlation_y.npy')

    # model_lasso = MultiTaskLasso()
    # print("Initialize Lasso")
    # model_lasso.fit(X_train_small, y_train_small)
    # print("Lasso fitted")

    # print('MAE Lasso: ', mean_absolute_error(y_val_small, model_lasso.predict(X_val_small)))

    R_small = np.copy(R[:N_TARGETS, :N_TARGETS])
    R_small[R_small < 0.8] = 0
    del R
    # R_small[np.abs(R_small) < 0.8] = 0

    maes = []
    for l1_reg in [0.01, 0.05, 0.1]:
        for gamma in [0.01, 0.05, 0.1]:
            print('MAE GFlasso l1_reg=', l1_reg, ' gamma=', gamma)
            coeffs, obj = gflasso_2010(X_train_small, y_train_small, R_small, l1_reg=l1_reg, gamma=gamma,
                                       iter_max=1e3)
            mae = mean_absolute_error(y_val_small, X_val_small.dot(coeffs))
            print('MAE GFlasso l1_reg=', l1_reg, ' gamma=', gamma, mae)
            maes.append(mae)
