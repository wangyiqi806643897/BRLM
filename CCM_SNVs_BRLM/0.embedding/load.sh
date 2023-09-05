export DATA_PATH=data.txt
export OUTPUT_PATH=data.h5

python load_embedding.py \
    --inputtext_path ${DATA_PATH}\
    --indexed_path ${OUTPUT_PATH}