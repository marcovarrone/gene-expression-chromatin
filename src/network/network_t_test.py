import pandas as pd
import numpy as np
import pybedtools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import time

from chiapet_to_sparse import ChiaPetInteractions


def get_bed_compatible_df(df):
    bed_df = df[['dgex_feature_id', 'chr', 'start', 'end', 'strand']].copy()
    bed_df.columns = ['dgex_id', 'chrom', 'chromStart', 'chromEnd', 'strand']

    bed_df['strand'] = bed_df['strand'].astype('object')
    coordinates = bed_df.replace({'strand': {-1: '-', 1: '+'}}, None)
    return coordinates


def extension(row, size):
    if row['strand'] == '+':
        new_value = row['chromStart'] - 1000
        if new_value < 0:
            new_value = 0
        row['chromStart'] = new_value
    elif row['strand'] == '-':
        row['chromEnd'] += 1000
    return row


def extend_to_promoter(bed, size=1000):
    pass
    # Requires file with chromosome size
    # bed.slop(l=size, s=True)

    # Alternative: do it manually
    if isinstance(bed, pd.DataFrame):
        return bed.apply(lambda row: extension(row, size), axis=1)
    elif isinstance(bed, pybedtools.BedTool):
        # bed_df = pd.read_csv(bed, sep='\t', names=['chrom', 'chromStart', 'chromEnd', 'strand'])
        bed_df = bed.to_dataframe(names=['chrom', 'chromStart', 'chromEnd', 'strand'])
        bed_df = bed_df.apply(lambda row: extension(row, size), axis=1)
        return pybedtools.BedTool.from_dataframe(bed_df)


def get_interacting_genes(genes, position, chromosome):
    return genes[(position > genes['chromStart']) & (position < genes['chromEnd']) & (genes['chrom'] == chromosome)]


def count_genes_interactions(interactions, bed, n_genes):
    gene_interactions = np.zeros((n_genes, n_genes))

    for i, interaction in interactions.iterrows():
        first_genes = get_interacting_genes(bed, interaction['index1'], interaction['chrom1'])
        second_genes = get_interacting_genes(bed, interaction['index2'], interaction['chrom2'])
        if len(first_genes) > 0 and len(second_genes) > 0:
            for _, first_gene in first_genes.iterrows():
                for _, second_gene in second_genes.iterrows():
                    # if first_gene['strand'] == second_gene['strand']:
                    gene_interactions[first_gene['dgex_id']][second_gene['dgex_id']] += 1
                    #gene_interactions[second_gene['dgex_id']][first_gene['dgex_id']] += 1
    return gene_interactions


def compute_interactions(df, region_interactions, save=True):
    try:
        return np.load('gene_interactions_HCT116_POLR2A.npy')
    except FileNotFoundError:

        genes_coord = get_bed_compatible_df(df)

        bed_w_promoter = extend_to_promoter(genes_coord, 1000)

        genes_interactions = count_genes_interactions(region_interactions, bed_w_promoter, len(genes_coord))

        if save:
            np.save('gene_interactions_HCT116_POLR2A.npy', genes_interactions)
        return genes_interactions


def compute_correlations(expression, save=True):
    try:
        return np.load('gene_correlations_dgex.npy')
    except FileNotFoundError:
        gene_correlations = np.corrcoef(expression)
        if save:
            np.save('gene_correlations_dgex.npy', gene_correlations)
        return gene_correlations


def build_interaction_matrix(first_genes, second_genes, n_genes):
    gene_interactions = np.zeros(shape=(n_genes, n_genes))
    gene_interactions[first_genes, second_genes] = 1
    # gene_interactions[second_genes, first_genes] = 1
    return gene_interactions


def plot_connected_distribution(gene_interactions, gene_correlations):
    interaction_idxs = np.nonzero(gene_interactions)

    correlations_interacting = gene_correlations[interaction_idxs]
    sns.distplot(correlations_interacting)
    return correlations_interacting


def plot_random_distribution(gene_interactions, gene_correlations):
    n_interacting_gene_couples = np.count_nonzero(gene_interactions)
    index_x, index_y = np.where(gene_interactions == 0)

    sampling_idxs = np.random.choice(np.arange(index_x.shape[0]), n_interacting_gene_couples, replace=False)

    correlations_non_interacting_sampled = gene_correlations[index_x[sampling_idxs], index_y[sampling_idxs]]

    sns.distplot(correlations_non_interacting_sampled)
    return correlations_non_interacting_sampled


def plot_distributions(gene_interactions, gene_correlations):
    correlations_interacting = plot_connected_distribution(gene_interactions, gene_correlations)
    correlations_non_interacting_sampled = plot_random_distribution(gene_interactions, gene_correlations)
    plt.show()
    return correlations_interacting, correlations_non_interacting_sampled


def genemania(df, expression):
    path = '/home/nanni/Projects/gexi-top/data/external/COMBINED.DEFAULT_NETWORKS.BP_COMBINING.txt'
    gene_pairs = pd.read_table(path)
    gene_meta = df.sort_values("dgex_feature_id")
    all_genemania_genes = set(gene_pairs.Gene_A.tolist() + gene_pairs.Gene_B.tolist())
    gene_meta_filtered = gene_meta[gene_meta.ensembl_id.isin(all_genemania_genes)].sort_values("dgex_feature_id")
    gene_meta_filtered["dgex_feature_id_filtered"] = np.arange(gene_meta_filtered.shape[0], dtype=int)

    genexp_filtered = expression[gene_meta_filtered.dgex_feature_id.values, :]

    gene_correlations = np.corrcoef(genexp_filtered)

    gene_pairs_filtered = gene_pairs[
        gene_pairs.Gene_A.isin(gene_meta_filtered.ensembl_id.values) & gene_pairs.Gene_B.isin(
            gene_meta_filtered.ensembl_id.values)]

    gene_pairs_idxs = gene_pairs_filtered.merge(gene_meta_filtered[['ensembl_id', 'dgex_feature_id_filtered']],
                                                left_on='Gene_A', right_on='ensembl_id')
    gene_pairs_idxs.drop(['ensembl_id', 'Gene_A'], inplace=True, axis=1)
    gene_pairs_idxs = gene_pairs_idxs.rename(columns={'dgex_feature_id_filtered': 'Gene_A'})
    gene_pairs_idxs = gene_pairs_idxs.merge(gene_meta_filtered[['ensembl_id', 'dgex_feature_id_filtered']],
                                            left_on='Gene_B', right_on='ensembl_id')
    gene_pairs_idxs.drop(['ensembl_id', 'Gene_B'], inplace=True, axis=1)
    gene_pairs_idxs = gene_pairs_idxs.rename(columns={'dgex_feature_id_filtered': 'Gene_B'})
    __n_samples = 9000

    corr_sample = gene_pairs_idxs.sample(__n_samples, replace=False)
    return corr_sample, len(gene_meta_filtered), gene_correlations

def main():
    bin_length = 10000000
    file_path = 'ENCSR000BZX_HCT116_POLR2A.bed'
    contact_matrix = ChiaPetInteractions(file_path, bin_length, different_chrs=False)
    from_genemania = False

    gene_meta = pd.read_csv('/home/nanni/Projects/gexi-top/data/processed/dgex_genes_with_coords.tsv', sep='\t', header=0)
    genexp = np.load('/home/nanni/Projects/gexi-top/data/processed/d-gex/bgedv2_GTEx_1000G_float64.npy')

    if from_genemania:
        gene_pairs_idxs, n_genes, gene_correlations = genemania(gene_meta, genexp)
        gene_interactions = build_interaction_matrix(gene_pairs_idxs.Gene_A.values, gene_pairs_idxs.Gene_B.values, n_genes)
    else:
        gene_interactions = compute_interactions(gene_meta, contact_matrix.interactions)
        expression = genexp[gene_meta['dgex_feature_id']]
        gene_correlations = compute_correlations(expression)

    correlations_interacting, correlations_non_interacting_sampled = plot_distributions(gene_interactions,
                                                                                        gene_correlations)

    print(ttest_ind(correlations_interacting, correlations_non_interacting_sampled, equal_var=False))


if __name__ == '__main__':
    main()
