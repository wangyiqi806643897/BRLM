########### conda activate pytorch1.9

echo "BioBert entities generation..."

export MAX_LENGTH=384
export DATA_PATH=data/case_control.hg19_multiannoID.txt
export OUTPUT_PATH=0.embedding/data.h5
export BATCH_SIZE=100

python 0.embedding/run_embedding.py \
    --model_name_or_path dmis-lab/biobert-base-cased-v1.1 \
    --max_seq_length  ${MAX_LENGTH} \
    --data_path ${DATA_PATH} \
    --output_path ${OUTPUT_PATH} \
    --batch_size ${BATCH_SIZE} \
    --pooling mean

echo "ResNet50 Training and Testing..."
python main.py

echo "Mutant genes in Class123 Enrichment..."
Rscript Class123_enrich.r

echo "Perturbation Score calculation..."
Rscript pfs_score.r

echo "Please check all results in res_CCM_Classify/ !"
