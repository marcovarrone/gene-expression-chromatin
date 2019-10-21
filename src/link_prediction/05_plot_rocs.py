import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    roc1 = np.load('rocs_mlp_topological.npy')
    #roc2 = np.load('rocs_mlp_graphsage_slim_n20_10_l50_8_d0.0_r0.01.npy')
    roc3 = np.load('rocs_mlp_graphsage_n20_10_l50_8_d0.0_r0.01.npy')


    rocs = np.vstack((roc1, roc3)).T
    df_rocs = pd.DataFrame(rocs)
    ax = sns.boxplot(data=df_rocs)
    ax.set_ylim(0.4, 1)
    plt.ylabel('ROC-AUC')
    plt.xlabel('methods')
    plt.show()