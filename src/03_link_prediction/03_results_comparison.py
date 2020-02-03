import argparse
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

boxplot_palette = ['C4', 'C3', 'C2', 'C0', 'C1']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MCF7')
    parser.add_argument('--chrs', type=int, nargs='*', default=-1)
    parser.add_argument('--metric', type=str, choices=['acc', 'roc', 'f1'], default='acc')
    parser.add_argument('--embs', nargs='*', type=str)
    parser.add_argument('--plot', type=str, default='box')
    #parser.add_argument('--classifier', type=str, choices=['mlp', 'lr'], default='mlp')
    parser.add_argument('--full', default=False, action='store_true')
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
    labels_no_name = []
    labels_name = []
    for model in args.embs:
        label = ''
        #ToDo: prevent hackfix
        #if 'primary' in model:
        #    label += 'hic_'
        #elif 'chr' in model:
        #    label += 'coexp_'
        if 'gae' in model:
            label += 'gae'
        elif 'pca' in model:
            label += 'pca'
        elif 'svd' in model:
            label += 'SVD'
        elif 'line' in model:
            label += 'line'

        elif 'ids' in model:
            label += 'ids'
        elif 'trivial' in model:
            label += 'trivial'
        elif 'distance' in model:
            label += 'Distance'
        elif 'node2vec' in model:
            label += 'node2vec'
        elif 'random' in model:
            label += 'Random'

        label_no_name = label
        if 'topological' in model:
            label_no_name += 'Topological\nmeasures'
        if 'hadamard' in model and 'topological' not in model and 'distance' not in model:
            label_no_name += '\n(Hadamard)'
        #if 'observed' in model:
        #    label += '_obs'
        if 'oe' in model:
            label += '_oe'
        #if 'all' in model:
        #    label += '_all'
        label_name = label
        if 'topological' in model:
            label_name += 'topological'
        if 'mlp' in model:
            label_name += '_mlp'
        if 'lr' in model:
            label_name += '_lr'
        if 'rf' in model:
            label_name += '_rf'
        if 'hadamard' in model:
            label_name += '_had'
        if 'nwhad' in model:
            label_name += '_nwhad'
        if 'avg' in model:
            label_name += '_avg'
        if 'sub' in model:
            label_name += '_sub'
        if 'l2' in model:
            label_name += '_l2'
        if 'concat' in model:
            label_name += '_cat'
        if 'nwavg' in model:
            label_name += '_nwavg'
        if 'nwl1' in model:
            label_name += '_nwl1'
        if 'nwl2' in model:
            label_name += '_nwl2'
        labels.append(label)
        labels_no_name.append(label_no_name)
        labels_name.append(label_name)

        if args.full:
            print('results/{}/chr_all/{}.pkl'.format(args.dataset,model.format(chrs, chrs)))
            with open('results/{}/chr_all/{}.pkl'.format(args.dataset, model.format(chrs, chrs)),
                      'rb') as file_load:
                results = pickle.load(file_load)
                scores.append(results[args.metric])
        elif type(chrs) == int:
            print('results/{}/chr_{:02d}/{}.pkl'.format(args.dataset, chrs, model.format(chrs, chrs)))
            with open('results/{}/chr_{:02d}/{}.pkl'.format(args.dataset, chrs, model.format(chrs, chrs)), 'rb') as file_load:
                results = pickle.load(file_load)
                scores.append(results[args.metric])
            #rocs.append(
            #    np.load('results/{}/chr_{}/{}.npy'.format(args.dataset, chrs, model)))
        elif type(chrs) == str:
            print('results/{}/chr_{}/{}.pkl'.format(args.dataset, chrs, model.format(chrs, chrs)))
            with open('results/{}/chr_{}/{}.pkl'.format(args.dataset, chrs, model.format(chrs, chrs)), 'rb') as file_load:
                results = pickle.load(file_load)
                scores.append(np.mean(results[args.metric]))
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
    print(np.mean(scores), np.std(scores))

    print(labels_name)
    if type(chrs) == int:
        scores = np.vstack(scores).T
        df_scores = pd.DataFrame(scores)
        ax = sns.boxplot(data=df_scores, palette=boxplot_palette)
        ax.set_ylim(0.4, 0.9)
    else:
        if not args.scatter:
            scores = np.vstack(scores).T
            df_scores = pd.DataFrame(scores)
            if args.plot == 'box':
                ax = sns.boxplot(data=df_scores, palette=boxplot_palette)
                ax.set_ylim(0.4, 0.8)
            else:
                ax = sns.barplot(data=df_scores, palette=boxplot_palette)
                ax.set_ylim(0.4, 0.8)
        else:
            for i, key in enumerate(scores_dict.keys()):
                plt.scatter([i] * len(scores_dict[key]), scores_dict[key], alpha=0.5)


    plt.ylabel('Accuracy' if args.metric == 'acc' else args.metric)
    plt.xticks(np.arange(len(args.embs)), labels_no_name)
    plt.grid(axis='y')
    # plt.xlabel('methods')
    if args.chrs == -1 or args.chrs == 'all':
        plt.savefig('plots/{}/{}_chr_all_{}.pdf'.format(args.dataset, args.metric, '_'.join(labels_name)), bbox_inches='tight', transparent=True)
    elif type(chrs) == int or len(chrs) == 1:
        plt.savefig('plots/{}/{}_chr_{:02d}_{}.pdf'.format(args.dataset, args.metric, args.chrs, '_'.join(labels_name)), bbox_inches='tight', transparent=True)
    else:
        plt.savefig('plots/{}/{}_chr_{}_{}{}.pdf'.format(args.dataset, args.metric, '_'.join(list(map(str, args.chrs))), '_'.join(labels_name),
                                                        '_scatter' if args.scatter else ''), bbox_inches='tight', transparent=True)
    plt.show()
