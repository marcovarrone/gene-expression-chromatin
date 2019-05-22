import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

from autoencoder_func import Autoencoder
from data_loader import X_train, X, OFFSET


def pca_plot(X_embedding, roles):
    number = len(np.unique(roles))
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, number)]

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_embedding)

    pca_result = pd.DataFrame(data=principal_components,
                              columns=['pc1', 'pc2'])
    pca_result_labeled = pd.concat([pca_result, pd.Series(roles, name='type')], axis=1)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    for role_label, color in zip(np.flip(np.unique(roles)), colors):
        print(role_label)
        indices_to_keep = pca_result_labeled['type'] == role_label
        print(color)
        ax.scatter(pca_result_labeled.loc[indices_to_keep, 'pc1']
                   , pca_result_labeled.loc[indices_to_keep, 'pc2']
                   , c=color
                   , s=10)
    #ax.legend(role_labels)
    ax.grid()
    #plt.savefig('autoencoder_pca.png')
    plt.show()

def target_landmark_roles():
    targets = np.repeat('target', 12320 - 970)
    landmarks = np.repeat('landmark', 970)
    return np.hstack((landmarks, targets))


if __name__ == '__main__':
    autoencoder = Autoencoder(X_train.shape[1], embedding_size=50, learning_rate=0.0001,
                              batch_norm=False, run_folder=None, save_model=True, offset=OFFSET)
    autoencoder.fit(X_train, batch_size=128, epochs=200)
    encoder = autoencoder.encoder
    X_embedding = encoder.predict(X, batch_size=128)

    with open('clusters/autoencoder_lr0.0001_bs128_o20000.pkl', 'rb') as file_first:
        kmeans_first = pickle.load(file_first)
        #roles = target_landmark_roles()
        roles = kmeans_first.labels_
        pca_plot(X_embedding, roles)
