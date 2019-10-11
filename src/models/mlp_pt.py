import argparse
import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchsummary import summary

np.random.seed(42)
torch.manual_seed(42)


class DatasetGM(Dataset):

    def __init__(self, X, Y, transform=None):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
        self.transform = transform
        print(X.shape, Y.shape)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sample = self.X[index]
        target = self.Y[index]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target


num_epochs = 1500
batch_size = 32
learning_rate = 1e-4

torch.set_num_threads(5)

os.environ["OMP_NUM_THREADS"] = "5"
os.environ["OPENBLAS_NUM_THREADS"] = "5"
os.environ["MKL_NUM_THREADS"] = "5"
os.environ["VECLIB_MAXIMUM_THREADS"] = "5"
os.environ["NUMEXPR_NUM_THREADS"] = "5"

# train = np.load('/home/varrone/Prj/gene-expression-chromatin/src/embedding/data/GM19238/X_train_59_0_normalized_0.2_0.2.npy')
# train = np.load('/home/varrone/Prj/gene-expression-chromatin/src/embedding/data/GSE92743/X_train_25000_0_normalized_0.2_0.2.npy')
# test = torch.FloatTensor(np.load('/home/varrone/Prj/gene-expression-chromatin/src/embedding/data/GSE92743/X_train_25000_0_normalized_0.2_0.2.npy'))

# dataset_train = DatasetGM('/home/varrone/Prj/gene-expression-chromatin/src/embedding/data/GM19238/X_train_59_0_normalized_0.2_0.2.npy')
# dataset_train = DatasetGM('/home/varrone/Prj/gene-expression-chromatin/src/embedding/data/GSE92743/X_train_25000_0_normalized_0.2_0.2.npy')
# dataset_train = DatasetGM('/home/varrone/Prj/gene-expression-chromatin/src/embedding/data/GSE92743/X_train_25000_0_normalized_0.2_0.2.npy')

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"


class MLP(object):

    def __init__(self, in_dim, out_dim, hidden_size, n_layers, activation=nn.Tanh):
        self.model = MLPNetwork(in_dim, out_dim, hidden_size, n_layers, activation)

    def fit(self, dataloader_train, validation_data=None, n_epochs=200, lr=1e-2, weight_decay=0.0):
        self.model.to(device)
        # dataloader_train.to(device)
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr)

        dur = []
        for epoch in range(n_epochs):
            if epoch >= 3:
                t0 = time.time()

            for data in dataloader_train:
                sample, target = data
                sample = sample.to(device)
                target = target.to(device)

                # ===================forward=====================
                output = self.model(sample)
                loss = criterion(output, target)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            # ===================log========================
            if validation_data:

                val_loss = self.evaluate(validation_data[0], validation_data[1])
                print("Epoch {:05d}/{:05d} | Time(s) {:.4f} | Loss {:.4f} | Val loss {:.4f}".format(
                    epoch,
                    n_epochs,
                    np.mean(dur),
                    loss.item(),
                    val_loss))
            else:
                print("Epoch {:05d}/{:05d} | Time(s) {:.4f} | Loss {:.4f}".format(
                    epoch,
                    n_epochs,
                    np.mean(dur),
                    loss.item()))

    def encode(self, data):
        return self.model.encoder(data)

    def evaluate(self, X, Y):
        X = torch.FloatTensor(X)
        Y = torch.FloatTensor(Y)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(X)
            loss = torch.nn.L1Loss()
            return loss(prediction, Y)


class MLPNetwork(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_size, n_layers, activation):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_feats, hidden_size))
        self.layers.append(activation())

        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(activation())
        self.layers.append(nn.Linear(hidden_size, out_feats))

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='GSE92743')
    parser.add_argument('-tr', '--training-set', type=str)
    parser.add_argument('-v', '--validation-set', type=str)
    parser.add_argument('-te', '--test-set', type=str)
    parser.add_argument('-l', '--landmarks', type=str, default='l1000')
    parser.add_argument('-r', '--random-landmarks', type=int)
    parser.add_argument('-n', '--n-iter', type=int, default=1)
    parser.add_argument('-s', '--n-samplings', type=int, default=1)
    parser.add_argument('-m', '--model', type=str, default='lr')

    parser.add_argument('--threshold', type=float, default=0.001)
    parser.add_argument('--n-jobs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    train = np.load('data/' + str(args.dataset) + '/' + str(args.training_set) + '.npy').T
    landmarks = np.load('landmarks/' + str(args.dataset) + '/' + str(args.landmarks) + '.npy')
    n_genes = train.shape[1]
    targets = np.arange(n_genes)
    targets = np.setdiff1d(targets, landmarks)

    X_train = torch.FloatTensor(train[:, landmarks])
    Y_train = torch.FloatTensor(train[:, targets])

    if args.validation_set:
        valid = np.load('data/' + str(args.dataset) + '/' + str(args.validation_set) + '.npy').T
        X_valid, Y_valid = torch.FloatTensor(valid[:, landmarks]), torch.FloatTensor(valid[:, targets])
        X_test, Y_test = X_valid, Y_valid
    if args.test_set:
        test = np.load('data/' + str(args.dataset) + '/' + str(args.test_set) + '.npy').T
        X_test, Y_test = torch.FloatTensor(test[:, landmarks]), torch.FloatTensor(test[:, targets])

    landmarks = np.load('landmarks/' + str(args.dataset) + '/' + str(args.landmarks) + '.npy')

    dataloader_train = DataLoader(DatasetGM(X_train, Y_train), batch_size=batch_size, shuffle=True)

    model = MLP(X_train.shape[1], 100, Y_train.shape[1])
    print(summary(model.model, X_train.shape, device="cpu"))
    model.fit(dataloader_train, X_valid, Y_valid, n_epochs=num_epochs, lr=learning_rate)
    loss_test = evaluate(model.model, X_test, Y_test)
    print("Test MAE {:.4f}".format(loss_test))
