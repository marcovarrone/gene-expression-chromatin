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

# Pipeline
![Alt text](pipeline.png)

# Guidelines
The order of execution of the pipeline is proprocessing -> network building -> link prediction.<br>

### Preprocessing
The pipeline is compatible with gene expression data downloaded from the [Xena Browser TCGA Hub](https://xenabrowser.net/datapages/?host=https%3A%2F%2Ftcga.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443).
The absolute path to the HiSeqV2 file is required in 01_gene_expression.py as the _--input_ parameter 
and a name for the dataset can be chosen to identify the specific dataset through the parameter _--dataset_.<br><br>

Since the original Hi-C datasets have different structure, we developed custom preprocessing scripts 
for the MCF-7 breast cancer (_Barutcu et al. (2015)_) and 22Rv1 prostate cancer (_Rhie et al. (2019)_) cell lines.

### Network building 
The first part of the pipeline is devoted to the building of the coexpression network from the gene expression data.<br>
If no chromosome is specified in the _--chr-src_ and _--chr-tgt_ parameters of 01_compute_coexpression.py, 
the coexpression will be computed for all the chromosomes at once for the single model setting and the 
intra+inter-chromosomal setting.<br><br>

In 02_coexpression_network.py, we can obtain networks based of which of the three analysis to perform:
+ different model for each chromosome: pass the _--single-chrom_ parameter and select the percentile for the threshold 
with _--perc-intra PERCENTILE_
+ a single model for all the intra-chromosomal co-expression: select the percentile for the threshold 
with _--perc-intra PERCENTILE_
+ a single model for all the chromosomes considering also inter-chromosomal co-expression: select the two different thresholds 
(for the intra-chromosomal and the inter-chromosomal co-expression) through the _--perc-intra PERCENTILE_ and  _--perc-inter PERCENTILE_ 
parameters.<br><br>

In 03_hic_gene_selection.py builds an Hi-C matrix of interactions between genes instead of bins.<br>
The _--genome-wide_ parameter allows to build an network that includes both intra- and inter-chromosomal interactions.<br><br>

The 05_chromatin_network.py script allows to build the Gene Chromatin Network by applying a threshold to the interactions.<br>
The principle for obtaining the three different types of network is equal to the co-expression network case.<br>
It is possible to select inter-chromosomal interactions with different resolution through the _--resolution-inter RESOLUTION_ parameter.

### Link prediction
The 01_matrix_factorization.py and 01_random_walk.py scripts allow to perform node embeddings based on the desired methods.<br>
The methods require the the [BioNEV library](https://github.com/xiangyue9607/BioNEV) to be installed.<br>
For both the scripts, the _--chromatin-network FILE_ specifies the name of the network to embed, present in the data/DATASET/chromatin_network folder after building the network.
Note that only the name is required (not the path) and without specifying the file format.<br><br>

The files 02_link_prediction_chromosome.py, 03_link_prediction_intra.py, 04_link_prediction_genomewide.py 
perform link prediction on, respectively, the separated chromosome, single intra-chromosomal model, 
single intrachromosomal+interchromosomal settings.<br>
The results of the prediction are saved in the _results_ folder as a pickle dictionary with the 
_acc, f1, roc, precision, recall_ keys depending on the type of performance measure.<br><br>

The 05_results_comparison.py script generates the plot for comparing the performances between the different models. <br>
With the _--embs_ parameter it is possible to select which type of models to compare by listing the name of the 
results used to produce the prediction results. It's important replace the chromosome number with "{}" to compare runs over multiple chromosomes.<br>
For example, to compare the topological measure method with node2vec, pass _--embs rf\_topological\_primary\_observed_ICE\_{}\_{}\_40000\_3.65\_avg\_l1 rf\_node2vec\_primary\_observed\_ICE\_{}\_{}\_40000\_3.59\_es16\_nw10\_wl80\_p1.0\_q1.0\_hadamard\_0.38_

## Example script

### Full pipeline
A script to automatically execute the whole pipeline for the single-chromosome and shared intra-chromosomal using the breast cancer dataset is provided through the *run_breast_cancer.sh* file.<br>
Both the expression and the Hi-C data are automatically downloaded, but the preprocessing of the Hi-C data requires the presence of the juicer tools jar available at https://github.com/aidenlab/juicer/wiki/Juicer-Tools-Quick-Start. The version used in the paper is *juicer_tools_1.13.02.jar*. To execute the pipeline using a different version it is sufficient to change the value of the JUICER_PATH variable.<br>
WARNING: verify that the machine used has the resources required to execute the pipeline.

### Building a coexpression network
An example script for generating a coexpression network from the initial RNA-seq data for chromosome 1 and saving the plot of both the coexpression matrix and the adjacency matrix of the coexpression network is provided through the *run_coexpression_breast_cancer.sh* file.<br><br>
By changing the value of the COEXP_PERCENTILE variable it is possible to select the threshold to consider when a pair of genes is coexpressed ror not.<br><br>
Since the threshold percentile is compute across all the chromosome. The method must compute the coexpression matrices for all the chromosomes first.

### Building a chromatin network
An example script for generating the chromatin network from the Hi-C data for chromosome 1 and saving the plot of both the Hi-C matrix and the adjacency matrix of the chromatin network is provided through the *run_chromatin_breast_cancer_sh* file.<br><br>
It is possible to change the resolution of the Hi-C matrix to extract the links from by changing the value of the RESOLUTION variable (IMPORTANT: choose a value for ORIGINAL_RESOLUTION that is the highest among the options avaiable from juicer that are lower or equal to RESOLUTION). Similarly to the coexpression case you can change the thresholding value thorugh the HIC_PERCENTILE variable.<br><br>
Since the threshold percentile is compute across all the chromosome. The method must compute the Hi-C matrices for all the chromosomes first.

### Predicting the coexpression using the node2vec embeddings
An example script for predicting the coexpression of chromosome 1 using node embedding extracted with the node2vec algorithm and combined into edge embeddings using the hadamard operator is provieded through the *run_node2vec_chr1.sh* file.<br><br>
The script contains all the variable required to control the characteristics of the networks to be used, the hyperaparameters of node2vec and the link prediction parameters.