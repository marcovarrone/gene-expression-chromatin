import os
import argparse
import pandas as pd
import numpy as np


def main(args):
    reads = pd.read_csv(args.expression_path, delimiter='\t', skiprows=2)
    reads = reads.set_index('Description')

    info = pd.read_csv(args.annotations_path, delimiter='\t')
    tissues = np.unique(info['SMTSD'])

    samples_reads_ids = reads.columns
    for tissue in tissues:
        print('Tissue', tissue)
        sample_ids = info[info['SMTSD'] == tissue]['SAMPID']
        sample_ids = np.intersect1d(sample_ids, samples_reads_ids)
        samples_tissue = reads[sample_ids]
        samples_tissue = samples_tissue.rename(columns={'Description': 'sample'})
        samples_tissue = np.log1p(samples_tissue)

        filename = tissue.lower().replace(' - ', '-').replace(' ', '_')
        samples_tissue.to_csv(os.path.join(args.output_path, filename+'.tsv'), sep='\t', index_label='sample')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--expression-path', type=str, default='/home/varrone/Data/GTEx/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct')
    parser.add_argument('--annotations-path', type=str, default='/home/varrone/Data/GTEx/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt')
    parser.add_argument('--output-path', type=str, default='/home/varrone/Data/GTEx/datasets_tpm')
    args = parser.parse_args()

    main(args)

