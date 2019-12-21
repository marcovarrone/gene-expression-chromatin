import argparse

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # ToDo: add description
    parser.add_argument('-d', '--dataset', type=str, default='MCF7')
    parser.add_argument('--method', type=str, default='line')
    #parser.add_argument('--embedding', type=str, default='primary_observed_KR_all_50000_50000_primary_observed_NONE_all_100000_100000_es8')
    #parser.add_argument('--embedding', type=str, default='primary_observed_KR_1_1_50000_50000_0.9073_es8')
    #parser.add_argument('--embedding', type=str, default='primary_observed_KR_1_1_50000_50000_0.9073_es8_h32_e50_lr0.01_do0.0')
    parser.add_argument('--embedding', type=str, default='primary_observed_KR_all_50000_50000_primary_observed_NONE_all_100000_100000_no_inter_es8_e10_lr0.01')
    #parser.add_argument('--embedding', type=str, default='/home/varrone/Repo/gae/emb_primary_observed_KR_all_50000_50000_primary_observed_NONE_all_100000_100000.txt')
    parser.add_argument('--all-regions', default=False, action='store_true')
    parser.add_argument('--numpy', default=True, action='store_true')
    parser.add_argument('--genes-chr', type=int, default=None)
    args = parser.parse_args()

    if args.numpy:
        embeddings = np.load(
            '/home/varrone/Prj/gene-expression-chromatin/src/link_prediction/embeddings/{}/{}/{}.npy'.format(
                args.dataset, args.method, args.embedding))

    else:
        with open(args.embedding) as f:
            node_num, emb_size = f.readline().split()

            embeddings = np.zeros((int(node_num), int(emb_size)))
            for line in f:
                vec = line.strip().split()
                node_id = int(vec[0])
                emb_arr = np.array(vec[1:], dtype=np.float64)
                emb_arr = emb_arr / np.linalg.norm(emb_arr)
                emb_arr[np.isnan(emb_arr)] = 0
                embeddings[node_id, :] = emb_arr
    adj = np.dot(embeddings, embeddings.T)

    plt.imshow(adj, cmap='Oranges')
    plt.show()
