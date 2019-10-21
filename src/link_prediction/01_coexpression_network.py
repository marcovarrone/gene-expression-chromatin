import numpy as np

if __name__ == '__main__':
    expression = np.load('/home/varrone/Prj/gene-expression-chromatin/src/chromatin/data/GM19238/rna/GM19238_chr11.npy')
    coexpression = np.corrcoef(expression)
    np.fill_diagonal(coexpression, 0)
    threshold = np.percentile(coexpression, 90)
    coexpression[coexpression < threshold] = 0
    coexpression[coexpression >= threshold] = 1

    np.save('data/GM19238/coexpression_'+str(90)+'.npy', coexpression)
