import cooler
import configparser
import os

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import matplotlib.style as style
plt.style.use('ggplot')

config = configparser.ConfigParser()
config.read('/home/varrone/config.ini')

def add_gene_coordinates(genes, gene_info):
    gene_info_filtered = genes.merge(gene_info, on='Gene_IDs')
    #gene_info_filtered['position'] = (gene_info_filtered['Gene start (bp)'] + gene_info_filtered['Gene end (bp)']) // 2
    return gene_info_filtered

def get_gene_bins(gene, offset, bin_size):
    tss = gene['Transcription start site (TSS)']
    tss_bin = offset + tss // bin_size
    start = tss_bin - 40000 // bin_size
    end = tss_bin + 40000 // bin_size
    return start, end

def hic_plot(c, gene_info):
    path = '/home/varrone/Prj/gene-expression-chromatin/src/preprocessing/GM19238/GSE63525_GM12878_insitu_primary_10kb_contact_matrix.npy'
    if os.path.exists(path):
        contact_matrix = np.load(path)
    else:
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
        np.save(path, contact_matrix)
    plt.imshow(np.rot90(np.log2(contact_matrix)), cmap="Oranges")
    plt.show()

def coregulation_plot(df_rna):
    df_rna_sorted = df_rna.sort_values('Transcription start site (TSS)')
    gene_exp = df_rna_sorted.iloc[:, 2:].values
    gene_exp_med = np.nanmedian(gene_exp, axis=0)
    gene_exp -= gene_exp_med
    coexp = np.corrcoef(gene_exp)
    plt.imshow(np.rot90(1 - coexp), cmap="RdBu") # 1 - coexp to invert the colormap
    plt.show()

if __name__ == '__main__':

    df = pd.read_csv(config['GM19238']['X'])
    gene_info = pd.read_csv('/home/varrone/Data/GM19238/GRCh37_gene_info.txt', '\t')
    gene_info = gene_info.drop_duplicates('Gene stable ID')
    gene_info = gene_info.rename(columns={'Gene stable ID': 'Gene_IDs'})
    df_rna = df.iloc[:, 3:df.columns.get_loc('RPKM_GM19257') + 1]
    gene_chr_tss = gene_info[gene_info['Chromosome/scaffold name'] == '11'][['Gene_IDs', 'Transcription start site (TSS)']]
    df_rna = add_gene_coordinates(gene_chr_tss, df_rna)

    hic_plot(cooler.Cooler('/home/varrone/Data/GM19238/GSE63525_GM12878_insitu_primary_10kb.cool'), gene_chr_tss)




