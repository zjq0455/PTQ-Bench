MODEL=llava-v1.5-7b

# run AWQ search (optional; we provided the pre-computed results)
CUDA_VISIBLE_DEVICES=0 python -m awq.entry --model_path $MODEL \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/$MODEL-w4-g128.pt

# generate real quantized weights (w4)
CUDA_VISIBLE_DEVICES=0 python -m awq.entry --model_path $MODEL \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/$MODEL-w4-g128.pt \
    --q_backend fake \
    --dump_fake ./fake/$MODEL-w4-g128
