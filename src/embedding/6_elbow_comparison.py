import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

pca = np.hstack(([0], np.load('elbow/GSE92743/elbow_pca_0.9_10_500.npy')))
autoencoder = np.hstack(
    ([0], np.load('elbow/GSE92743/elbow_autoencoder_50_e50_lr0.0001_bs128_bn_20000_0_normalized_10_500.npy')))
graphsage = np.hstack(([0], np.load(
    'elbow/GSE92743/elbow_graphsage_autoencoder_50_e50_lr0.0001_bs128_bn_20000_0_normalized_10_500.npy')))

k_range_low = np.arange(0, 100, 10)
# k_range_high = np.arange(100, 1000, 200)
k_range_high = np.arange(100, 500, 50)
k_range = np.hstack((k_range_low, k_range_high))

plt.plot(k_range, pca)
plt.plot(k_range, autoencoder)
plt.plot(k_range, graphsage)

plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.ylabel('Explained variance')

plt.xlabel('N. of clusters')
plt.legend(('PCA', 'Autoencoder', 'GraphSAGE'))
plt.show()
#plt.savefig('elbow/GSE92743/comparison.png')
