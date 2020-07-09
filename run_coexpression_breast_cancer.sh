DATASET=breast_cancer
EXPRESSION_PATH=../../data/${DATASET}/HiSeqV2
COEXP_PERCENTILE=90.0
CHROMOSOME=1

mkdir -p data/${DATASET}

cd data/${DATASET}
if [ ! -f "$EXPRESSION_PATH" ]; then
  wget https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz
  gunzip HiSeqV2.gz
fi


cd ../../src/preprocessing
python3 01_gene_expression.py --input $EXPRESSION_PATH --dataset $DATASET

cd ../network_building
python3 01_compute_coexpression.py --dataset $DATASET  --save-plot --save-coexp
python3 02_coexpression_network.py --dataset $DATASET --chr-src $CHROMOSOME --chr-tgt $CHROMOSOME --perc-intra $COEXP_PERCENTILE --save-matrix --save-plot