import pandas as pd

if __name__ == '__main__':

    tcga = pd.read_csv('/home/varrone/Data/PRAD/HiSeqV2', delimiter='\t')
    print(tcga.shape)
    tcga = tcga[(tcga == 0).sum(axis=1) <= tcga.shape[1] * 0.2]
    print(tcga.shape)

    gene_info = pd.read_csv('/home/varrone/Data/MCF7/GRCh37_p13_gene_info.txt', delimiter='\t')


    tcga = gene_info.merge(tcga, right_on='sample', left_on='Gene name',)
    tcga = tcga.drop('sample', axis=1)
    tcga_pos = tcga[tcga['Strand'] == 1]
    tcga_neg = tcga[tcga['Strand'] == -1]
    tcga_pos = tcga_pos.groupby(['Gene name']).min()
    tcga_neg = tcga_neg.groupby(['Gene name']).max()


    tcga = pd.concat([tcga_neg, tcga_pos])
    print(tcga.shape)
    tcga = tcga.groupby(['Gene name']).max()
    print(tcga.shape)

    #scaler = StandardScaler()
    #tcga_exp_scaled = scaler.fit_transform(tcga_exp)
    #tcga.to_csv('/home/varrone/Data/PRAD/TCGA_PRAD_GRCh37_p13.csv')
    pass
