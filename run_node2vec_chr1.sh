DATASET=breast_cancer
RESOLUTION=40000
COEXP_PERCENTILE=90.0
HIC_PERCENTILE=80.0
CHROMOSOME=1

NUM_WALKS=10
WALK_LEN=80
P=1
Q=1
WINDOW=10
EMBEDDING_SIZE=16

AGGREGATION=hadamard


cd src/link_prediction
python3 01_random_walk.py --dataset $DATASET --chromatin-network observed_${CHROMOSOME}_${CHROMOSOME}_${RESOLUTION}_${HIC_PERCENTILE} --emb-size $EMBEDDING_SIZE --num-walks $NUM_WALKS --walk-len $WALK_LEN --p $P --q $Q --window $WINDOW --save-emb --task none
python3 02_link_prediction_chromosome.py --dataset $DATASET --chr-src $CHROMOSOME --chr-tgt $CHROMOSOME --method node2vec --chromatin-network-name observed_${CHROMOSOME}_${CHROMOSOME}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators $AGGREGATION --coexp-thr $COEXP_PERCENTILE