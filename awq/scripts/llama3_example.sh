
CUDA_VISIBLE_DEVICES=5 python -m awq.entry --model_path $model \
--w_bit 3 --q_group_size 128 \
--run_awq --dump_awq awq_cache/$model-w3-g128.pt


CUDA_VISIBLE_DEVICES=5 python -m awq.entry --model_path $model \
    --tasks wikitext \
    --w_bit 3 --q_group_size 128 \
    --load_awq awq_cache/$model-w3-g128.pt \
    --q_backend fake --dump_fake ./fake/$model-w3-g128