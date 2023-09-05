########### conda activate pytorch1.9

echo "BioBert entities generation..."

while read id
do
        export MAX_LENGTH=384
        export DATA_PATH="TCGA_maf/"$id"/merge_anno.hg19_multianno_none.txt"
        export OUTPUT_PATH="0.embedding/"$id"_data.h5"
        export BATCH_SIZE=1

        python 0.embedding/run_embedding.py \
            --model_name_or_path dmis-lab/biobert-base-cased-v1.1 \
            --max_seq_length  ${MAX_LENGTH} \
            --data_path ${DATA_PATH} \
            --output_path ${OUTPUT_PATH} \
            --batch_size ${BATCH_SIZE} \
            --pooling mean
done<TCGA_maf/name

echo "ResNet50 Training and Testing..."
while read id
do
	python main.py $id
done < res_TCGA_vectors/name

echo "Please check all results in res_TCGA_vectors/ !"
