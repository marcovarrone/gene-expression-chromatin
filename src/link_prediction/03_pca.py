import os
import argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import spectral_embedding



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MCF7')
    parser.add_argument('--chr', type=int, default=11)
    parser.add_argument('--interactions', type=str)
    parser.add_argument('--threshold', type=int, default=90)
    parser.add_argument('--n-components', type=int, default=8)
    args = parser.parse_args()

    hic = np.load('data/{}/interactions/interactions_{}.npy'.format(args.dataset,args.interactions))

    pca = PCA(n_components=args.n_components)
    embeddings = pca.fit_transform(hic)

    if not os.path.exists('embeddings/pca/'):
        os.makedirs('embeddings/pca/')

    np.save('embeddings/{}/pca/{}_es{}.npy'.format(args.dataset, args.interactions,  args.n_components), embeddings)

