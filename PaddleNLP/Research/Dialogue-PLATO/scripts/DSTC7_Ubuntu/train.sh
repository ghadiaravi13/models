#!/bin/bash
set -ux

SAVE_DIR=outputs/DSTC7_Ubuntu
VOCAB_PATH=model/Bert/vocab.txt
DATA_DIR=data/DSTC7_Ubuntu
INIT_CHECKPOINT=model/PLATO
DATA_TYPE=multi
USE_VISUALDL=false
BATCH_SIZE=$1
# preferred bs is 6

# CUDA environment settings.
export CUDA_VISIBLE_DEVICES=0

# Paddle environment settings.
export FLAGS_fraction_of_gpu_memory_to_use=0.1
export FLAGS_eager_delete_scope=True
export FLAGS_eager_delete_tensor_gb=0.0

python -u \
    ./preprocess.py \
    --vocab_path $VOCAB_PATH \
    --data_dir $DATA_DIR \
    --data_type $DATA_TYPE

if [[ "$USE_VISUALDL" = true ]]; then
    visualdl --logdir=$SAVE_DIR/summary --port=8083 --host=`hostname` &
    VISUALDL_PID=$!
fi

python -u \
    ./run.py \
    --do_train true \
    --vocab_path $VOCAB_PATH \
    --data_dir $DATA_DIR \
    --data_type $DATA_TYPE \
    --batch_size $BATCH_SIZE \
    --valid_steps 2000 \
    --num_type_embeddings 2 \
    --use_discriminator true \
    --num_epoch 15 \
    --lr 1e-5 \
    --save_checkpoint false \
    --save_summary $USE_VISUALDL \
    --init_checkpoint $INIT_CHECKPOINT \
    --save_dir $SAVE_DIR

if [[ $USE_VISUALDL = true ]]; then
    kill $VISUALDL_PID
fi