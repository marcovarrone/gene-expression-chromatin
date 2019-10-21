import numpy as np

if __name__ == '__main__':
    interactions = np.load('/home/varrone/Prj/gene-expression-chromatin/src/chromatin/data/GM19238/GSE63525_GM12878_insitu_primary_10kb_contact_matrix.npy')
    threshold = np.percentile(interactions, 80)
    interactions[interactions < threshold] = 0
    interactions[interactions >= threshold] = 1

    np.save('data/GM19238/interactions_'+str(80)+'.npy', interactions)