import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA

from umap import UMAP


def target_landmark_roles():
    targets = np.repeat('target', 12320 - 970)
    landmarks = np.repeat('landmark', 970)
    return np.hstack((landmarks, targets))


def roles_from_landmarks(landmarks):
    roles = np.array(['target'] * 12320, dtype='object')
    roles[landmarks] = 'landmark'
    return roles


def pca_plot(embedding, roles=None, colors=None, filename_save=None):
    if roles is None:
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
        if role_label == 'landmark':
            ax.scatter(pca_result_labeled.loc[indices_to_keep, 'pc1']
                       , pca_result_labeled.loc[indices_to_keep, 'pc2']
                       , c=color
                       , s=30)
        else:
            ax.scatter(pca_result_labeled.loc[indices_to_keep, 'pc1']
                       , pca_result_labeled.loc[indices_to_keep, 'pc2']
                       , c=color
                       , s=30, alpha=0.5)
    ax.legend(*np.unique(roles))
    ax.grid()
    if filename_save:
        plt.savefig(str(filename_save))
    plt.show()


def tsne_plot(embedding, landmarks=None, colors=None, filename_save=None, perplexity=30):
    if landmarks is None:
        roles = target_landmark_roles()
    else:
        roles = roles_from_landmarks(landmarks)
    number = len(np.unique(roles))
    cmap = plt.get_cmap('RdYlGn')
    if not colors:
        colors = [cmap(i) for i in np.linspace(0, 1, number)]

    tsne = TSNE(n_jobs=8, verbose=1, n_components=2, random_state=42, perplexity=perplexity)
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
        if role_label == 'landmark':
            ax.scatter(final_tsne_df.loc[indices_to_keep, 'component1']
                       , final_tsne_df.loc[indices_to_keep, 'component2']
                       , c=color
                       , s=30)
        else:
            ax.scatter(final_tsne_df.loc[indices_to_keep, 'component1']
                       , final_tsne_df.loc[indices_to_keep, 'component2']
                       , c=color
                       , s=30, alpha=0.5)

    ax.legend(role_labels)
    ax.grid()
    if filename_save:
        plt.savefig(str(filename_save))
    plt.show()


def umap_plot(embedding, roles=None, colors=None, filename_save=None, n_neighbors=15):
    if roles is None:
        roles = target_landmark_roles()
    number = len(np.unique(roles))
    cmap = plt.get_cmap('RdYlGn')
    if not colors:
        colors = [cmap(i) for i in np.linspace(0, 1, number)]

    umap = UMAP(n_neighbors=n_neighbors)
    umap_results = umap.fit_transform(embedding)
    umap_df = pd.DataFrame(data=umap_results,
                           columns=['component1', 'component2'])
    final_umap_df = pd.concat([umap_df, pd.Series(roles, name='type')], axis=1)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Component 1', fontsize=15)
    ax.set_ylabel('Component 2', fontsize=15)
    ax.set_title('2 component t-SNE', fontsize=20)
    role_labels = ['target', 'landmark']
    for role_label, color in zip(role_labels, colors):
        indices_to_keep = final_umap_df['type'] == role_label
        if role_label == 'landmark':
            ax.scatter(final_umap_df.loc[indices_to_keep, 'component1']
                       , final_umap_df.loc[indices_to_keep, 'component2']
                       , c=color
                       , s=30)
        else:
            ax.scatter(final_umap_df.loc[indices_to_keep, 'component1']
                       , final_umap_df.loc[indices_to_keep, 'component2']
                       , c=color
                       , s=30, alpha=0.5)

    ax.legend(role_labels)
    ax.grid()
    if filename_save:
        plt.savefig(str(filename_save))
    plt.show()
