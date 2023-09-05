export MAX_LENGTH=384
export DATA_PATH=case_vs_control.hg19_multianno2.txt
export OUTPUT_PATH=data.h5
export BATCH_SIZE=1

python run_embedding.py \
    --model_name_or_path dmis-lab/biobert-base-cased-v1.1 \
    --max_seq_length  ${MAX_LENGTH} \
    --data_path ${DATA_PATH} \
    --output_path ${OUTPUT_PATH} \
    --batch_size ${BATCH_SIZE} \
    --pooling mean
