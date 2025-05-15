MODEL=VILA1.5-7b

# # run AWQ search (optional; we provided the pre-computed results)
# # Note: vila checkpoints are now stored in 3 parts.
# # only llm folder will be quantized
CUDA_VISIBLE_DEVICES=0 python -m awq.entry --model_path /share/$MODEL/llm \
    --w_bit 2 --q_group_size 128 --vila-15 \
    --run_awq --dump_awq /share/awq_cache/$MODEL-w2-g128.pt

# # generate real quantized weights (w4)
CUDA_VISIBLE_DEVICES=0 python -m awq.entry --model_path /share/$MODEL/llm \
    --w_bit 2 --q_group_size 128 --vila-15 \
    --load_awq /share/awq_cache/$MODEL-w2-g128.pt \
    --q_backend fake --dump_fake /share/LLM_EVAL/AWQ/$MODEL-w2/llm
