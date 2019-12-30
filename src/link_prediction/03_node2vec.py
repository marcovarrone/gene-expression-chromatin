import argparse

from link_prediction.utils import set_n_threads, generate_embedding

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MCF7')
    parser.add_argument('--interactions', type=str,
                        default=None,
                        required=True)
    parser.add_argument('--coexpression', type=str, default=None)
    parser.add_argument('--emb-size', type=int, default=8)
    parser.add_argument('--n-jobs', type=int, default=10)
    parser.add_argument('--num-walks', type=int, default=10)
    parser.add_argument('--walk-len', type=int, default=80)
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--q', type=float, default=1.0)
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=None)

    parser.add_argument('--all-regions', default=False, action='store_true')
    parser.add_argument('--force', default=True, action='store_true')
    parser.add_argument('--method', default='node2vec', type=str, choices=['node2vec', 'struc2vec'])
    parser.add_argument('--weighted', type=str, default='False')
    parser.add_argument('--save-emb', default=False, action='store_true')
    parser.add_argument('--task', type=str, default='none')
    parser.add_argument('--genes-chr', default=False, action='store_true')

    args = parser.parse_args()

    set_n_threads(args.n_jobs)

    if args.coexpression:
        args.folder = 'coexpression'
        args.name = args.coexpression
    else:
        args.folder = 'interactions'
        args.name = args.interactions

    # ToDo: handle better the adding of the threshold term in the interactions path
    # (and its absence in gene_chr path)
    interactions_path = './data/{}/{}/{}_{}{}.npy'.format(
        args.dataset, args.folder, args.folder, args.name, '_' + str(args.threshold) if args.threshold else '')

    emb_path = '{}{}_es{}_nw{}_wl{}_p{}_q{}'.format(args.name,
                                                    '_' + str(args.threshold) if args.threshold else '', args.emb_size,
                                                    args.num_walks, args.walk_len, args.p, args.q)

    command = 'bionev --input data/{}/{}/{}{}.edgelist '.format(args.dataset, args.folder, args.name, '_' + str(
        args.threshold) if args.threshold else '') + \
              '--output ./embeddings/{}/{}/{}.txt '.format(args.dataset, args.method.lower(), emb_path) + \
              '--method {} --task {} '.format(args.method, args.task) + \
              '--dimensions {} '.format(args.emb_size) + \
              '--number-walks {} '.format(args.num_walks) + \
              '--walk-length {} '.format(args.walk_len) + \
              '--p {} --q {} '.format(args.p, args.q) + \
              '--window-size {} '.format(args.window) + \
              ('--weighted True' if args.weighted == 'True' else '')

    generate_embedding(args, emb_path, interactions_path, command)
