import argparse
import time

import networkx as nx
import numpy as np
import scipy.sparse as sps
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch.conv import SAGEConv
from sklearn.preprocessing import StandardScaler


class GraphSAGE(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 n_hidden,
                 n_layers,
                 activation,
                 input_dropout,
                 dropout,
                 feature_dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g
        self.input_dropout = input_dropout
        self.dropout = dropout

        # input layer
        self.layers.append(
            SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=feature_dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=feature_dropout, activation=activation))
        # output layer
        self.layers.append(
            SAGEConv(n_hidden, out_feats, aggregator_type, feat_drop=feature_dropout,
                     activation=None))  # activation None

    def forward(self, features):
        h = features
        h = F.dropout(h, self.input_dropout, training=self.training)
        for layer in self.layers[:-1]:
            h = layer(self.g, h)
            h = F.dropout(h, self.dropout, training=self.training)
        h = self.layers[-1](self.g, h)
        return h

    def embedding(self, features):
        h = features
        for i, layer in enumerate(self.layers[:-1]):
            h = layer(self.g, h)
        return h


class GraphSAGEWrapper:
    def __init__(self,
                 in_feats,
                 out_feats,
                 graph,
                 n_hidden,
                 n_layers,
                 activation=F.tanh,
                 input_dropout=0.0,
                 dropout=0.0,
                 feature_dropout=0.0,
                 aggregator_type='mean',
                 cuda=False,
                 self_loop=True,
                 wandb=False):

        self.cuda = cuda
        self.wandb = wandb
        if self.wandb:
            wandb.config.update({"architecture": "graphsage_layers_dropout",
                                 "n_hidden": n_hidden,
                                 "n_layers": n_layers,
                                 "activation": activation.__class__.__name__,
                                 "input_dropout": input_dropout,
                                 "dropout": dropout,
                                 "aggregator_type": aggregator_type,
                                 "self_loop": self_loop
                                 })

        g_nx = nx.from_numpy_array(graph, create_using=nx.DiGraph)

        # add self loop
        if self_loop:
            g_nx.remove_edges_from(g_nx.selfloop_edges())
        g_nx.add_edges_from(zip(g_nx.nodes(), g_nx.nodes()))
        g = DGLGraph()
        g.from_networkx(g_nx)
        # g = DGLGraph(g)

        # normalization
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        if cuda:
            norm = norm.cuda()
        g.ndata['norm'] = norm.unsqueeze(1)
        self.g = g
        self.model = GraphSAGE(g,
                               in_feats,
                               out_feats,
                               n_hidden,
                               n_layers,
                               activation,
                               input_dropout,
                               dropout,
                               feature_dropout,
                               aggregator_type)

    def fit(self, X, Y, train_mask, val_mask=None, n_epochs=200, lr=1e-2, weight_decay=0.0):
        if self.wandb:
            self.wandb.config.update({"epochs": n_epochs, "learning_rate": lr})
        X = torch.FloatTensor(X)
        Y = torch.FloatTensor(Y)
        n_edges = self.g.number_of_edges()
        if self.cuda:
            self.model.cuda()
        loss_fcn = torch.nn.L1Loss()

        # use optimizer
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)

        # initialize graph
        dur = []
        if self.wandb:
            self.wandb.watch(self.model)
        for epoch in range(n_epochs):
            self.model.train()
            if epoch >= 3:
                t0 = time.time()
            # forward
            prediction = self.model(X)
            loss = loss_fcn(prediction[train_mask], Y[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            if val_mask is not None:
                val_loss = self.evaluate(X, Y, val_mask)
                print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Val loss {:.4f} | "
                      "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                                    val_loss, n_edges / np.mean(dur) / 1000))
                if self.wandb:
                    self.wandb.log({"Validation Loss": val_loss})
            else:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | ETputs(KTEPS) {:.2f}".format(
                    epoch,
                    np.mean(dur),
                    loss.item(),
                    n_edges / np.mean(dur) / 1000))

    def embeddings(self, features):
        return self.model.embedding(features).detach().numpy()

    def test(self, X, Y, mask):
        loss = self.evaluate(X, Y, mask)
        if self.wandb:
            self.wandb.log({"Test Loss": loss})
        return loss

    def evaluate(self, X, Y, mask):
        X = torch.FloatTensor(X)
        Y = torch.FloatTensor(Y)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(X)
            loss = torch.nn.L1Loss()
            return loss(prediction[mask], Y[mask])


# nx.set_node_attributes(g_nx, dict(enumerate(features)))


def main(args):
    # load and preprocess dataset
    # data = load_data(args)
    features = np.load(
        '/home/varrone/Prj/gene-expression-chromatin/src/embedding/data/GSE92743/X_train_10000_0_normalized_0.2_0.2.npy')
    features = StandardScaler().fit_transform(features)
    # features = features.T[[þ[
    train_mask = np.random.choice(features.shape[0], int(features.shape[0] * 0.8), replace=False)
    val_mask = np.setdiff1d(np.arange(features.shape[0]), train_mask)
    print(features.shape, train_mask.shape, val_mask.shape)
    # features = features.T
    # features, test = dataset[train_idxs], dataset[test_idxs]
    # print(features.shape)
    features = torch.FloatTensor(features)

    # test = np.load('test_0_0.2.npy')
    # test = test.T
    # print(test.shape)
    # test = torch.FloatTensor(test)
    # labels = torch.LongTensor(data.labels)
    # train_mask = torch.ByteTensor(data.train_mask)
    # val_mask = torch.ByteTensor(data.val_mask)
    # test_mask = torch.ByteTensor(data.test_mask)
    in_feats = features.shape[1]
    # n_classes = data.num_labels
    # n_edges = data.graph.number_of_edges()
    '''print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           #train_mask.sum().item(),
           #val_mask.sum().item(),
           #test_mask.sum().item()))'''

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        # labels = labels.cuda()
        # train_mask = train_mask.cuda()
        # val_mask = val_mask.cuda()
        # test_mask = test_mask.cuda()

    # graph preprocess and calculate normalization factor
    adjacency = sps.load_npz('/home/varrone/Data/GSE92743_CMap/graphs/GSE92743_CMap_genemania.npz').todense()
    print(adjacency.shape)
    adjacency[adjacency < 0.0001] = 0
    adjacency[adjacency >= 0.0001] = 1
    # A = np.load('GSE63525_GM12878_insitu_primary_10kb_11_rna_norm.npy')

    # A[A > 0] = 1
    # A = A[train_idxs[:,None], train_idxs]

    # create GCN model
    model = GraphSAGE(adjacency,
                      in_feats,
                      in_feats,
                      args.n_hidden,
                      args.n_layers,
                      F.relu,
                      args.dropout,
                      args.aggregator_type
                      )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    # register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=50,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-6,
                        help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--aggregator-type", type=str, default="mean",
                        help="Aggregator type: mean/gcn/pool/lstm")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)
