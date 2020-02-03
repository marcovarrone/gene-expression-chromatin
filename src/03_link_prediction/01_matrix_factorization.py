import argparse

from link_prediction.utils import set_n_threads, generate_embedding

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MCF7')
    #parser.add_argument('--interactions', type=str, default='primary_observed_KR_all_50000_50000_primary_observed_NONE_all_50000_50000')
    parser.add_argument('--interactions', type=str, default='primary_observed_KR_all_50000_50000_2.74_1_primary_observed_NONE_all_500000_500000_9.0_1')
    parser.add_argument('--coexpression', type=str, default=None)
    parser.add_argument('--emb-size', type=int, default=16)
    parser.add_argument('--n-jobs', type=int, default=10)
    parser.add_argument('--all-regions', default=False, action='store_true')
    parser.add_argument('--chr-gene-idxs', type=int, default=None)  # ToDo: remove
    parser.add_argument('--genes-chr', default=False, action='store_true')
    parser.add_argument('--force', default=False, action='store_true')
    parser.add_argument('--method', default='SVD', type=str, choices=['Laplacian', 'SVD', 'HOPE'])
    parser.add_argument('--weighted', type=str, default='False')
    parser.add_argument('--save-emb', default=False, action='store_true')
    parser.add_argument('--task', type=str, default='link-prediction')
    args = parser.parse_args()

    set_n_threads(args.n_jobs)

    if args.coexpression:
        args.folder = 'coexpression'
        args.name = args.coexpression
    else:
        args.folder = 'interactions'
        args.name = args.interactions

    interactions_path = './data/{}/{}/{}_{}.npz'.format(
        args.dataset, args.folder, args.folder, args.name)

    emb_path = '{}_es{}'.format(args.interactions, args.emb_size)

    command = 'bionev --input data/{}/interactions/{}.edgelist '.format(args.dataset, args.interactions) + \
              '--output ./embeddings/{}/{}/{}.txt '.format(args.dataset, args.method.lower(), emb_path) + \
              '--method {} --task {} '.format(args.method, args.task) + \
              '--dimensions {} '.format(args.emb_size) + \
              ('--weighted True' if args.weighted == 'True' else '')

    generate_embedding(args, emb_path, interactions_path, command)