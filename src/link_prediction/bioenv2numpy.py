import numpy as np
from bionev.main import load_embedding

if __name__ == '__main__':
    #for chrom in range(1,10):
    chrom = 2
    emb_dict = load_embedding('/home/varrone/Repo/gae/emb_1.txt'.format(chrom))
    emb = np.zeros((len(emb_dict.keys()),6))
    for i in range(emb.shape[0]):
        emb[i, :] = emb_dict[str(i)]
    np.save('/home/varrone/Prj/gene-expression-chromatin/src/link_prediction/embeddings/embeddings_chr_{:02d}_gae_70.npy'.format(chrom), emb)
