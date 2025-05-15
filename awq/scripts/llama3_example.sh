
CUDA_VISIBLE_DEVICES=5 python -m awq.entry --model_path /share/$model \
--w_bit 3 --q_group_size 128 \
--run_awq --dump_awq /share/awq_cache/$model-w3-g128.pt


CUDA_VISIBLE_DEVICES=5 python -m awq.entry --model_path /share/$model \
    --tasks wikitext \
    --w_bit 3 --q_group_size 128 \
    --load_awq /share/awq_cache/$model-w3-g128.pt \
    --q_backend fake --dump_fake /share/LLM_EVAL/AWQ/$model-w3-g128