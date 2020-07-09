JUICER_PATH=../../juicer_tools_1.13.02.jar
DATASET=breast_cancer
HIC_PATH=https://hicfiles.s3.amazonaws.com/external/barutcu/MCF-7.hic
ORIGINAL_RESOLUTION=10000
RESOLUTION=40000
HIC_PERCENTILE=80.0
CHROMOSOME=1

mkdir -p data/${DATASET}

cd src/preprocessing
python3 02_hic_juicer.py --input $HIC_PATH --juicer-path $JUICER_PATH --dataset $DATASET --resolution $ORIGINAL_RESOLUTION --window $RESOLUTION

cd ../network_building
python3 03_hic_gene_selection.py --dataset $DATASET --type observed --resolution $RESOLUTION --save-matrix --save-plot
python3 04_chromatin_network.py --dataset $DATASET --chr-src $CHROMOSOME --chr-tgt $CHROMOSOME --type observed --resolution $RESOLUTION --type-inter observed --resolution-inter $RESOLUTION --perc-intra $HIC_PERCENTILE --save-matrix --save-plot
