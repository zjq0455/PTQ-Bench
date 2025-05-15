MODEL=llama-7b

# run AWQ search (optional; we provided the pre-computed results)
python -m awq.entry --model_path /dataset/llama-hf/$MODEL \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/$MODEL-w4-g128.pt

# evaluate the AWQ quantize model (simulated pseudo quantization)
python -m awq.entry --model_path /dataset/llama-hf/$MODEL \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/$MODEL-w4-g128.pt \
    --q_backend fake

