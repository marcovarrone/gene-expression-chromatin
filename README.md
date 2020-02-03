# Co-Expression Network Inference from Chromatin Conformation Data through Graph Embedding

The repository contains the source code for the link prediction problem of the paper:<br>
Varrone, Nanni et al., "Co-Expression Network Inference from Chromatin Conformation Data through Graph Embedding".<br><br>

## Abstract
### Motivation
The relationship between gene coexpression and chromatin conformation is of great biological
interest. Thanks to high-throughput chromosome conformation capture technologies (Hi-C), researchers
are gaining insights on the tri-dimensional organization of the genome. Given the high complexity of Hi-C
data and the difficult definition of gene coexpression networks, the development of proper computational
tools to investigate such relationship is rapidly gaining the interest of the research community. One of the
most fascinating questions in this context is how chromatin topology correlates with coexpression profiles
of genes and which physical interaction patterns are most predictive of coexpression relationships.

### Results
To address these questions, we developed a computational framework for the prediction of
coexpression networks from chromatin conformation data. We first define a gene chromatin interaction
network where each gene is associated to its physical interaction profile; then we apply two graph
embedding techniques to extract a low-dimensional vector representation of each gene from the interaction
network; finally, we train a Random Forest classifier on pairs of gene embeddings to predict if they are
coexpressed.<br>
Both graph embedding techniques outperform previous methods based on manually designed topological
features, highlighting the need for more advanced strategies to encode chromatin information. We also
establish that the most recent technique, based on random walks, is superior. Overall, our results
demonstrate that chromatin conformation and gene regulation share a non-linear relationship and that
gene topological embeddings encode relevant information, which could be used also for downstream
analysis.