#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1
N=$1
export MODEL_NAME="transformer-zh2en-word30K"

mpirun -np 2 \
    -H localhost:2 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python -m src.bin.translate \
    --model_name $MODEL_NAME \
    --source_path "/home/weihr/NMT_DATA_PY3/1.34M/unittest/MT0$1/src.txt" \
    --model_path "./save/$MODEL_NAME.best.tpz" \
    --config_path "./configs.yaml" \
    --batch_size 20 \
    --beam_size 5 \
    --saveto "./result/$MODEL_NAME.MT0$1.txt" \
    --source_bpe_codes "" \
    --use_gpu --multi_gpu