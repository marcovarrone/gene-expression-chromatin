import numpy as np
from embedding.plots import tsne_plot, umap_plot
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':

    emb = np.load('/home/varrone/Prj/gene-expression-chromatin/src/link_prediction/embeddings/MCF7/struc2vec/primary_oe_NONE_2_2_10000_40000_sum_0.0_es8.npy')
    emb = StandardScaler().fit_transform(emb)
    tsne_plot(emb, landmarks=np.arange(emb.shape[0]), gradient=True)