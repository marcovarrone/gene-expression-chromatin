import cooler
import configparser

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import matplotlib.style as style
plt.style.use('ggplot')

config = configparser.ConfigParser()
config.read('/home/varrone/config.ini')

def add_gene_coordinates(genes, gene_info):
    gene_info_filtered = genes.merge(gene_info, left_on='Gene_IDs', right_on='Gene stable ID')
    gene_info_filtered['position'] = (gene_info_filtered['Gene start (bp)'] + gene_info_filtered['Gene end (bp)']) // 2
    return gene_info_filtered

def get_gene_bins(gene, offset, bin_size):
    tss = gene['Transcription start site (TSS)']
    tss_bin = offset + tss // bin_size
    start = tss_bin - 40000 // bin_size
    end = tss_bin + 40000 // bin_size
    return start, end

def hic_plot(c, gene_info):
    contact_matrix = np.zeros((gene_info.shape[0], gene_info.shape[0]))
    print(gene_info.shape[0], "genes to be processed")
    for i, (idx1, gene1) in enumerate(gene_info.iterrows()):
        start1, end1 = get_gene_bins(gene1, c.offset('11'), 10000)
        if i % 10 == 0:
            print("Gene", i, "completed")
        for j, (idx2, gene2) in enumerate(gene_info.iterrows()):
            start2, end2 = get_gene_bins(gene2, c.offset('11'), 10000)
            mat = c.matrix(balance=False)[start1:end1, start2:end2]
            value = np.median(np.median(mat))
            contact_matrix[i, j] = value
    np.save('/home/varrone/Prj/gene-expression-chromatin/src/preprocessing/GM19238/GSE63525_GM12878_insitu_primary_10kb_contact_matrix.npy', contact_matrix)

if __name__ == '__main__':

    df = pd.read_csv(config['GM19238']['X'])
    gene_info = pd.read_csv('/home/varrone/Data/GM19238/GRCh37_gene_info.txt', '\t')
    gene_info = gene_info.drop_duplicates('Gene stable ID')
    df_rna = df.iloc[:, 3:df.columns.get_loc('RPKM_GM19257') + 1]
    gene_info = add_gene_coordinates(pd.DataFrame(df['Gene_IDs'], columns=['Gene_IDs']), gene_info[gene_info['Chromosome/scaffold name'] == '11'])
    gene_info = gene_info.sort_values('Transcription start site (TSS)')
    hic_plot(cooler.Cooler('/home/varrone/Data/GM19238/GSE63525_GM12878_insitu_primary_10kb.cool'), gene_info)




