# BioBERT Embedding

- 安装对应包
  
  - pip install transformers
  
  - pip -install h5py

要从BioBERT中获取上下文化表征，请运行以下命令。
输出以hdf5格式保存，因此需要首先安装h5py包（“pip -install h5py”）。
提供了一个示例输入文本（`data.txt`），其中包含每行的生物医学概念。

1. 执行emb.sh文件，不需要修改
   
   - 执行sh emb.sh
   
   - DATA_PATH 文件名 例如"data.txt"
   
   - OUTPUT_PATH 生成h5文件 例如"data.h5"

```bash
export MAX_LENGTH=384
export DATA_PATH=data.txt
export OUTPUT_PATH=data.h5
export BATCH_SIZE=64

python run_embedding.py \
    --model_name_or_path dmis-lab/biobert-base-cased-v1.1 \
    --max_seq_length  ${MAX_LENGTH} \
    --data_path ${DATA_PATH} \
    --output_path ${OUTPUT_PATH} \
    --batch_size ${BATCH_SIZE} \
    --pooling mean
```

## Load Embeddings

2. 读取表征
   
   - sh load_emb.sh
   
   - DATA_PATH 文件名 例如"data.txt"
   
   - OUTPUT_PATH 生成h5文件 例如"data.h5"

```bash
export DATA_PATH=data.txt
export OUTPUT_PATH=data.h5

python load_embedding.py \
    --inputtext_path ${DATA_PATH}\
    --indexed_path ${OUTPUT_PATH}
```

读取表征向量可以仿照load_embedding中的main函数：

- 改data_args.indexed_path为自己的h5路径，“data.h5"

- 改data_args.inputtext_path为自己的data路径，“data.txt"

## Result

```
The number of keys in h5: 2048
entity_name = Lohmann Selected Leghorn
embedding = [2.77513593e-01  2.03759596e-02  1.59252986e-01 ...  7.65920877e-02  2.49284402e-01 -1.48969248e-01]
```
