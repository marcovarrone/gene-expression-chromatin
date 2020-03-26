import pandas as pd
import numpy as np
import pdb

if __name__ == '__main__':
    reads = pd.read_csv('/home/varrone/Data/GTEx/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct', delimiter='\t', skiprows=2)
    reads = reads.set_index('Description')

    info = pd.read_csv('/home/varrone/Data/GTEx/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt', delimiter='\t')
    tissues = np.unique(info['SMTSD'])


    samples_reads_ids = reads.columns
    for tissue in tissues:
        print('Tissue', tissue)
        sample_ids = info[info['SMTSD'] == tissue]['SAMPID']
        sample_ids = np.intersect1d(sample_ids, samples_reads_ids)
        samples_tissue = reads[sample_ids]
        samples_tissue = samples_tissue.rename(columns={'Description': 'sample'})
        samples_tissue = np.log1p(samples_tissue)
        samples_tissue.to_csv('/home/varrone/Data/GTEx/datasets_tpm/{}.tsv'.format(tissue.lower().replace(' - ', '-').replace(' ', '_')), sep='\t', index_label='sample')

