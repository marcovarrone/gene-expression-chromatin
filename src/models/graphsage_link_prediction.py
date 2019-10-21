import stellargraph as sg
import keras
import numpy as np
import tensorflow as tf
from keras import optimizers, losses, metrics, Model
from stellargraph.data import EdgeSplitter
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.mapper import GraphSAGELinkGenerator, GraphSAGENodeGenerator
from wandb.keras import WandbCallback
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import networkx as nx

seed = 42
tf.set_random_seed(seed)
np.random.seed(seed)

def train(
        G,
        layer_size,
        num_samples,
        batch_size: int = 100,
        num_epochs: int = 10,
        learning_rate: float = 0.001,
        dropout: float = 0.0,
        study=None,
        trial=None,
        wandb=None
):
    """
    Train the GraphSAGE model on the specified graph G
    with given parameters.
    Args:
        G: NetworkX graph file
        layer_size: A list of number of hidden units in each layer of the GraphSAGE model
        num_samples: Number of neighbours to sample at each layer of the GraphSAGE model
        batch_size: Size of batch for inference
        num_epochs: Number of epochs to train the model
        learning_rate: Initial Learning rate
        dropout: The dropout (0->1)
    """
    if type(layer_size) != list:
        layer_size = [int(layer_size)]
    if type(num_samples) != list:
        num_samples = [int(num_samples)]
    batch_size = int(batch_size)
    num_epochs = int(num_epochs)

    # Split links into train/test
    print(
        "Using '{}' method to sample negative links".format("global")
    )
    non_edges = np.array(list(nx.non_edges(G)))
    print(len(non_edges))
    # From the original graph, extract E_test and the reduced graph G_test:
    edge_splitter_test = EdgeSplitter(G)
    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G, and obtain the
    # reduced graph G_test with the sampled links removed:
    G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
        p=0.05,
        keep_connected=True,
        method="global",
        seed=seed
    )

    # From G_test, extract E_train and the reduced graph G_train:
    edge_splitter_train = EdgeSplitter(G_test, G)
    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G_test, and obtain the
    # further reduced graph G_train with the sampled links removed:
    G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
        p=0.05,
        keep_connected=True,
        method="global",
        seed=seed
    )

    # G_train, edge_ds_train, edge_labels_train will be used for model training
    # G_test, edge_ds_test, edge_labels_test will be used for model testing

    # Convert G_train and G_test to StellarGraph objects (undirected, as required by GraphSAGE) for ML:
    G_train = sg.StellarGraph(G_train, node_features="feature")
    G_test = sg.StellarGraph(G_test, node_features="feature")

    # Mapper feeds link data from sampled subgraphs to GraphSAGE model
    # We need to create two mappers: for training and testing of the model
    train_gen = GraphSAGELinkGenerator(
        G_train, batch_size, num_samples, name="train", seed=seed
    ).flow(edge_ids_train, edge_labels_train, shuffle=True)

    test_gen = GraphSAGELinkGenerator(
        G_test, batch_size, num_samples, name="test",seed=seed
    ).flow(edge_ids_test, edge_labels_test, shuffle=True)

    # GraphSAGE model
    graphsage = GraphSAGE(
        layer_sizes=layer_size, generator=train_gen, bias=True, dropout=dropout
    )

    # Construct input and output tensors for the link prediction model
    x_inp, x_out = graphsage.build()

    # Final estimator layer
    prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
    )(x_out)

    # Stack the GraphSAGE and prediction layers into a Keras model, and specify the loss
    model = Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=optimizers.Adam(lr=learning_rate),
        loss=losses.binary_crossentropy,
        metrics=[metrics.binary_accuracy],
    )

    # Evaluate the initial (untrained) model on the train and test set:
    init_train_metrics = model.evaluate_generator(train_gen)
    init_test_metrics = model.evaluate_generator(test_gen)

    print("\nTrain Set Metrics of the initial (untrained) model:")
    for name, val in zip(model.metrics_names, init_train_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    print("\nTest Set Metrics of the initial (untrained) model:")
    for name, val in zip(model.metrics_names, init_test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    # Train model
    print("\nTraining the model for {} epochs...".format(num_epochs))

    callbacks = [EarlyStopping(monitor='val_binary_accuracy', patience=20, verbose=1, mode='max', restore_best_weights=True),
                   ReduceLROnPlateau(monitor='val_binary_accuracy', factor=0.5,patience=10,verbose=1, mode='max')]

    if wandb:
        callbacks.append(WandbCallback())
    if study:
        study.keras_callback(trial, objective_name='val_binary_accuracy')
    history = model.fit_generator(
        train_gen, epochs=num_epochs, validation_data=test_gen, verbose=2, shuffle=False,
        callbacks=callbacks
    )

    # Evaluate and print metrics
    train_metrics = model.evaluate_generator(train_gen)
    test_metrics = model.evaluate_generator(test_gen)

    print("\nTrain Set Metrics of the trained model:")
    for name, val in zip(model.metrics_names, train_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    print("\nTest Set Metrics of the trained model:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    # Save the trained model
    '''save_str = "_n{}_l{}_d{}_r{}".format(
        "_".join([str(x) for x in num_samples]),
        "_".join([str(x) for x in layer_size]),
        dropout,
        learning_rate,
    )
    model.save("graphsage_link_pred" + save_str + ".h5")'''

    # Extract embedding
    x_inp_src = x_inp[0::2]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
    node_ids = np.arange(G.number_of_nodes())
    node_gen = GraphSAGENodeGenerator(sg.StellarGraph(G, node_features="feature"), batch_size, num_samples, seed=seed).flow(
        node_ids)
    node_embeddings = embedding_model.predict_generator(node_gen, workers=4, verbose=1)
    return model, node_embeddings


def test(G, model, batch_size: int = 100):
    """
    Load the serialized model and evaluate on a random balanced subset of all links in the graph.
    Note that the set of links the model is evaluated on may contain links from the model's training set.
    To avoid this, set the seed of the edge splitter to the same seed as used for link splitting in train()
    Args:
        G: NetworkX graph file
        model_file: Location of Keras model to load
        batch_size: Size of batch for inference
    """
    '''print("Loading model from ", model_file)
    model = keras.models.load_model(
        model_file, custom_objects={"MeanAggregator": MeanAggregator}
    )'''

    # Get required input shapes from model
    num_samples = [
        int(model.input_shape[ii + 1][1] / model.input_shape[ii][1])
        for ii in range(1, len(model.input_shape) - 1, 2)
    ]

    edge_splitter_test = EdgeSplitter(G)
    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G, and obtain the
    # reduced graph G_test with the sampled links removed:
    G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
        p=0.05, method="global",seed=seed
    )

    # Convert G_test to StellarGraph object (undirected, as required by GraphSAGE):
    G_test = sg.StellarGraph(G_test, node_features="feature")

    # Generator feeds data from (source, target) sampled subgraphs to GraphSAGE model
    test_gen = GraphSAGELinkGenerator(
        G_test, batch_size, num_samples, name="test", seed=seed
    ).flow(edge_ids_test, edge_labels_test)

    # Evaluate and print metrics
    test_metrics = model.evaluate_generator(test_gen)

    print("\nTest Set Evaluation:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    return test_metrics[1]