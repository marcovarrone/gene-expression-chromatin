JUICER_PATH=../../juicer_tools_1.13.02.jar
DATASET=breast_cancer
EXPRESSION_PATH=../../data/${DATASET}/HiSeqV2
HIC_PATH=https://hicfiles.s3.amazonaws.com/external/barutcu/MCF-7.hic
ORIGINAL_RESOLUTION=10000
RESOLUTION=40000
COEXP_PERCENTILE=90.0
HIC_PERCENTILE=80.0
EMBEDDING_SIZE=16


mkdir -p data/${DATASET}

cd data/${DATASET}
if [ ! -f "$EXPRESSION_PATH" ]; then
  wget https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz
  gunzip HiSeqV2.gz
fi



cd ../../src/preprocessing
python3 01_gene_expression.py --input $EXPRESSION_PATH --dataset $DATASET
python3 02_hic_juicer.py --input $HIC_PATH --juicer-path $JUICER_PATH --dataset $DATASET --resolution $ORIGINAL_RESOLUTION --window $RESOLUTION

cd ../network_building
python3 01_compute_coexpression.py --dataset $DATASET --save-plot --save-coexp
python3 02_coexpression_network.py --dataset $DATASET --perc-intra $COEXP_PERCENTILE --save-matrix --save-plot
python3 03_hic_gene_selection.py --dataset $DATASET --type observed --resolution $RESOLUTION --save-matrix --save-plot
python3 04_chromatin_network.py --dataset $DATASET --type observed --resolution $RESOLUTION --type-inter observed --resolution-inter $RESOLUTION --perc-intra $HIC_PERCENTILE --save-matrix --save-plot

cd ../link_prediction
for i in {1..22}
do
  python3 01_matrix_factorization.py --dataset $DATASET --chromatin-network observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --emb-size $EMBEDDING_SIZE --save-emb --task none
  python3 01_random_walk.py --dataset $DATASET --chromatin-network observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --emb-size $EMBEDDING_SIZE --save-emb --task none
done

for i in {1..22}
do
  python3 02_link_prediction_chromosome.py --dataset $DATASET --chr-src $i --chr-tgt $i --method random --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --coexp-thr $COEXP_PERCENTILE --classifier random --force
  python3 02_link_prediction_chromosome.py --dataset $DATASET --chr-src $i --chr-tgt $i --method distance --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators hadamard --coexp-thr $COEXP_PERCENTILE
  python3 02_link_prediction_chromosome.py --dataset $DATASET --chr-src $i --chr-tgt $i --method topological --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators avg l1 --coexp-thr $COEXP_PERCENTILE
  python3 02_link_prediction_chromosome.py --dataset $DATASET --chr-src $i --chr-tgt $i --method svd --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators hadamard --coexp-thr $COEXP_PERCENTILE
  python3 02_link_prediction_chromosome.py --dataset $DATASET --chr-src $i --chr-tgt $i --method node2vec --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators hadamard --coexp-thr $COEXP_PERCENTILE
done

for i in {1..22}
do
  python3 02_link_prediction_chromosome.py --dataset $DATASET --chr-src $i --chr-tgt $i --method random --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --coexp-thr $COEXP_PERCENTILE --classifier random --force
  python3 02_link_prediction_chromosome.py --dataset $DATASET --chr-src $i --chr-tgt $i --method distance --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --test
  python3 02_link_prediction_chromosome.py --dataset $DATASET --chr-src $i --chr-tgt $i --method topological --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators avg l1 --coexp-thr $COEXP_PERCENTILE --test
  python3 02_link_prediction_chromosome.py --dataset $DATASET --chr-src $i --chr-tgt $i --method svd --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --test
  python3 02_link_prediction_chromosome.py --dataset $DATASET --chr-src $i --chr-tgt $i --method node2vec --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --test
done

python3 01_matrix_factorization.py --dataset $DATASET --chromatin-network observed_all_40000_${HIC_PERCENTILE} --emb-size $EMBEDDING_SIZE --save-emb --task none
python3 01_random_walk.py --dataset $DATASET --chromatin-network observed_all_${RESOLUTION}_${HIC_PERCENTILE} --emb-size $EMBEDDING_SIZE --save-emb --task none --num-walks 55 --p 1.5 --q 4.5 --walk-len 55 --window 15

python3 03_link_prediction_intra.py --dataset $DATASET --method distance --type observed --coexp-thr $COEXP_PERCENTILE --type observed --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --verbose
python3 03_link_prediction_intra.py --dataset $DATASET --method topological --type observed --aggregators avg l1 --coexp-thr $COEXP_PERCENTILE --type observed --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --verbose
python3 03_link_prediction_intra.py --dataset $DATASET --method svd --type observed  --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --type observed --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --verbose
python3 03_link_prediction_intra.py --dataset $DATASET --method node2vec --type observed  --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --type observed --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --num-walks 55 --p 1.5 --q 4.5 --walk-len 55 --window 15 --verbose

python3 03_link_prediction_intra.py --dataset $DATASET --method distance --type observed --coexp-thr $COEXP_PERCENTILE --type observed --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --test
python3 03_link_prediction_intra.py --dataset $DATASET --method topological --type observed --aggregators avg l1 --coexp-thr $COEXP_PERCENTILE --type observed --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --test
python3 03_link_prediction_intra.py --dataset $DATASET --method svd --type observed  --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --type observed --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --test
python3 03_link_prediction_intra.py --dataset $DATASET --method node2vec --type observed  --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --type observed --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --num-walks 55 --p 1.5 --q 4.5 --walk-len 55 --window 15   --test


