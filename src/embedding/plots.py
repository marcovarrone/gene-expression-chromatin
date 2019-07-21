import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA

def target_landmark_roles():
    targets = np.repeat('target', 12320 - 970)
    landmarks = np.repeat('landmark', 970)
    return np.hstack((landmarks, targets))


def pca_plot(embedding, roles=None, colors=None, filename_save=None):
    if not roles:
        roles = target_landmark_roles()
    number = len(np.unique(roles))
    cmap = plt.get_cmap('RdYlGn')
    if not colors:
        colors = [cmap(i) for i in np.linspace(0, 1, number)]

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(embedding)

    pca_result = pd.DataFrame(data=principal_components,
                              columns=['pc1', 'pc2'])
    pca_result_labeled = pd.concat([pca_result, pd.Series(roles, name='type')], axis=1)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    for role_label, color in zip(np.flip(np.unique(roles)), colors):
        indices_to_keep = pca_result_labeled['type'] == role_label
        ax.scatter(pca_result_labeled.loc[indices_to_keep, 'pc1']
                   , pca_result_labeled.loc[indices_to_keep, 'pc2']
                   , c=color
                   , s=10)
    ax.legend(*np.unique(roles))
    ax.grid()
    if filename_save:
        plt.savefig(str(filename_save))
    plt.show()


def tsne_plot(embedding, roles=None, colors=None, filename_save=None):
    if not roles:
        roles = target_landmark_roles()
    number = len(np.unique(roles))
    cmap = plt.get_cmap('RdYlGn')
    if not colors:
        colors = [cmap(i) for i in np.linspace(0, 1, number)]

    tsne = TSNE(n_jobs=8, verbose=1, n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embedding)
    tsne_df = pd.DataFrame(data=tsne_results,
                           columns=['component1', 'component2'])
    final_tsne_df = pd.concat([tsne_df, pd.Series(roles, name='type')], axis=1)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Component 1', fontsize=15)
    ax.set_ylabel('Component 2', fontsize=15)
    ax.set_title('2 component t-SNE', fontsize=20)
    role_labels = ['target', 'landmark']
    for role_label, color in zip(role_labels, colors):
        indices_to_keep = final_tsne_df['type'] == role_label
        ax.scatter(final_tsne_df.loc[indices_to_keep, 'component1']
                   , final_tsne_df.loc[indices_to_keep, 'component2']
                   , c=color
                   , s=50)
    ax.legend(role_labels)
    ax.grid()
    if filename_save:
        plt.savefig(str(filename_save))
    plt.show()