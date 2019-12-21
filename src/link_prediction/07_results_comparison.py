import argparse
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='GM19238')
    parser.add_argument('--chrs', type=int, nargs='*', default=11)
    parser.add_argument('--metric', type=str, choices=['acc', 'roc', 'f1'], default='acc')
    parser.add_argument('--embs', nargs='*', type=str,)
    #parser.add_argument('--classifier', type=str, choices=['mlp', 'lr'], default='mlp')
    parser.add_argument('--save-fig', default=False, action='store_true')
    parser.add_argument('--scatter', default=False, action='store_true')
    args = parser.parse_args()
    scores = []
    # fig, ax = plt.subplots()
    scores_dict = defaultdict(list)
    if type(args.chrs) == list and len(args.chrs) == 1:
        args.chrs = args.chrs[0]

    if args.chrs == -1:
        chrs = list(range(1, 23))
    else:
        chrs = args.chrs

    labels = []
    for model in args.embs:
        label = ''
        #ToDo: prevent hackfix
        if 'gae' in model:
            label += 'gae'
        elif 'pca' in model:
            label += 'pca'
        elif 'svd' in model:
            label += 'svd'
        elif 'line' in model:
            label += 'line'
        elif 'topological' in model:
            label += 'topological'
        elif 'trivial' in model:
            label += 'trivial'
        elif 'distance' in model:
            label += 'distance'
        elif 'node2vec' in model:
            label += 'node2vec'
        elif 'random' in model:
            label += 'random'
        if 'observed' in model:
            label += '_obs'
        if 'oe' in model:
            label += '_oe'
        if 'all' in model:
            label += '_all'
        labels.append(label)

        if type(chrs) == int:
            print('results/{}/chr_{:02d}/{}.pkl'.format(args.dataset, chrs, model.format(chrs, chrs)))
            with open('results/{}/chr_{:02d}/{}.pkl'.format(args.dataset, chrs, model.format(chrs, chrs)), 'rb') as file_load:
                results = pickle.load(file_load)
                scores.append(results[args.metric])
            #rocs.append(
            #    np.load('results/{}/chr_{}/{}.npy'.format(args.dataset, chrs, model)))
        else:
            scores_chrs = []
            for chrom in chrs:
                #roc_chr = np.load(
                #    'results/{}/chr_{}/{}.npy'.format(args.dataset, chrom, model))
                with open('results/{}/chr_{:02d}/{}.pkl'.format(args.dataset, chrom, model.format(chrom, chrom)), 'rb') as file_load:
                    results = pickle.load(file_load)
                    score = np.mean(results[args.metric])
                    print(chrom, score)
                #roc_mean = np.mean(roc_chr)
                    scores_chrs.append(score)
            scores.append(scores_chrs)
            scores_dict[model] = scores_chrs

    if type(chrs) == int:
        scores = np.vstack(scores).T
        df_scores = pd.DataFrame(scores)
        ax = sns.boxplot(data=df_scores)
        ax.set_ylim(0.4, 0.8)
    else:
        if not args.scatter:
            scores = np.vstack(scores).T
            df_scores = pd.DataFrame(scores)
            ax = sns.boxplot(data=df_scores)
            ax.set_ylim(0.4, 0.8)
        else:
            for i, key in enumerate(scores_dict.keys()):
                plt.scatter([i] * len(scores_dict[key]), scores_dict[key], alpha=0.5)

    plt.ylabel(args.metric)
    plt.xticks(np.arange(len(args.embs)), labels)
    # plt.xlabel('methods')
    if args.chrs == -1:
        plt.savefig('plots/{}/{}_chr_all_{}.png'.format(args.dataset, args.metric, '_'.join(labels)))
    elif type(chrs) == int or len(chrs) == 1:
        plt.savefig('plots/{}/{}_chr_{:02d}_{}.png'.format(args.dataset, args.metric, args.chrs, '_'.join(labels)))
    else:
        plt.savefig('plots/{}/{}_chr_{}_{}{}.png'.format(args.dataset, args.metric, '_'.join(list(map(str, args.chrs))), '_'.join(labels),
                                                        '_scatter' if args.scatter else ''))
    plt.show()
