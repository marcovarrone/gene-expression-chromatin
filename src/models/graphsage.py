import configparser
import os

import keras
import pandas as pd
import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.mapper import GraphSAGELinkGenerator, GraphSAGENodeGenerator

from models.model import ModelNN

config = configparser.ConfigParser()
config.read('/home/varrone/config.ini')


# ToDo: implement Tensorboard tracking

class GraphSAGELinkPredictor(ModelNN):

    def __init__(self, graph, node_features, p_train=0.1, p_test=0.1, layer_sizes=None,
                 num_samples=None, dropout=0.3, optimizer=keras.optimizers.Adam, learning_rate=1e-4,
                 loss=keras.losses.binary_crossentropy, metric="acc", activation="relu",
                 edge_embedding_method="hadamard", batch_size=10, patience=10, checkpoint_every=0, save_model=False,
                 run_folder=None, embedding_representation=''):
        self.graph = graph
        self.node_features = node_features
        self.p_train = p_train
        self.p_test = p_test
        if layer_sizes is None:
            layer_sizes = [50, 50]
        self.layer_sizes = layer_sizes
        if num_samples is None:
            num_samples = [80, 10]
        self.num_samples = num_samples
        assert len(self.layer_sizes) == len(self.num_samples)

        self.dropout = dropout
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.metric = metric
        self.save_model = save_model
        self.emb_repr = embedding_representation

        for n in self.graph:
            self.graph.nodes[n]['feature'] = node_features[n]

        self.embeddings = None
        self.batch_size = batch_size
        self.edge_embedding_method = edge_embedding_method

        self._setup_graph(p_test, activation, batch_size)
        super().__init__(patience, checkpoint_every, save_model, run_folder)

    def _setup_graph(self, p_test, activation, batch_size):
        self.G = sg.StellarGraph(self.graph, node_features='feature')

        # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G,
        # obtain the reduced graph G_test with the sampled links removed:
        if p_test > 0:
            # Define an edge splitter on the original graph G:
            edge_splitter_test = EdgeSplitter(self.graph)
            G_test, self.edge_ids_test, self.edge_labels_test = edge_splitter_test.train_test_split(
                p=self.p_test, method="global", keep_connected=True
            )

            # Define an edge splitter on the reduced graph G_test:
            edge_splitter_train = EdgeSplitter(G_test)

            self.G_test = sg.StellarGraph(G_test, node_features="feature")
        else:
            edge_splitter_train = EdgeSplitter(self.graph)

        # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G_test,
        # obtain the reduced graph G_train with the sampled links removed:
        G_train, self.edge_ids_train, self.edge_labels_train = edge_splitter_train.train_test_split(
            p=self.p_train, method="global", keep_connected=True
        )

        self.G_train = sg.StellarGraph(G_train, node_features="feature")

        self.train_gen = GraphSAGELinkGenerator(self.G_train, batch_size, self.num_samples).flow(
            self.edge_ids_train, self.edge_labels_train, shuffle=True
        )
        if self.p_test > 0:
            self.test_gen = GraphSAGELinkGenerator(self.G_test, batch_size, self.num_samples).flow(
                self.edge_ids_test, self.edge_labels_test
            )
        else:
            self.test_gen = None

        graphsage = GraphSAGE(
            layer_sizes=self.layer_sizes, generator=self.train_gen, bias=True, dropout=self.dropout
        )

        self.x_inp, self.x_out = graphsage.build()

        self.prediction = link_classification(
            output_dim=1, output_act=activation, edge_embedding_method=self.edge_embedding_method
        )(self.x_out)

    def _build_model(self):

        model = keras.Model(inputs=self.x_inp, outputs=self.prediction)

        model.compile(
            optimizer=self.optimizer(lr=self.learning_rate),
            loss=self.loss,
            metrics=[self.metric],
        )

        self.output_model = model
        return model

    def fit(self, epochs=15, callbacks=None, verbose=1,
            use_multiprocessing=True, workers=10, force_train=False):
        self.epochs = epochs

        if force_train or not self._load_model():
            print("Start training of model " + str(self))
            callbacks = self._add_callbacks(callbacks)

            self.model.fit_generator(
                self.train_gen,
                callbacks=callbacks,
                epochs=epochs,
                validation_data=self.test_gen,
                verbose=verbose,
                use_multiprocessing=use_multiprocessing,
                workers=workers,
            )

            if self.save_model:
                self._save_model(self.test_gen)

        self.embeddings = self._generate_embedding(self.x_inp, self.x_out)



    def _generate_embedding(self, x_inp, x_out):
        x_inp_src = x_inp[0::2]
        x_out_src = x_out[0]

        embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
        node_features = pd.DataFrame(self.node_features)
        node_ids = node_features.index
        node_gen = GraphSAGENodeGenerator(self.G, self.batch_size, self.num_samples).flow(node_ids)

        emb = embedding_model.predict_generator(node_gen, workers=4, verbose=1)
        return emb[:, 0, :]

    def get_params(self, deep=True):
        return {"layer_sizes": str(self.layer_sizes),
                "n_samples": str(self.num_samples), "dropout": str(self.dropout)}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            print(parameter, value)
            setattr(self.model, parameter, value)
        return self

    def __str__(self):
        name = 'graphsage'
        name += '_l'
        for layer in self.layer_sizes:
            name += '_' + str(layer)
        name += '_ns'
        for samples in self.num_samples:
            name += '_' + str(samples)
        name += '_e' + str(self.epochs)
        name += '_d' + str(self.dropout)
        name += '_lr' + str(self.learning_rate)
        name += '_bs' + str(self.batch_size)
        name += '_' + str(self.edge_embedding_method)
        name += '_' + str(self.emb_repr)
        return name
