MODELS=("Llama-2-70b-hf")


for model in "${MODELS[@]}"
do

    # run AWQ search (optional; we provided the pre-computed results)
    CUDA_VISIBLE_DEVICES=6 python -m awq.entry --model_path $model \
        --w_bit 2 --q_group_size 128 \
        --run_awq --dump_awq ./awq_cache/$model-w2-g128.pt

done


for model in "${MODELS[@]}"
do

    # evaluate the AWQ quantize model (simulated pseudo quantization)
    CUDA_VISIBLE_DEVICES=2,7 python -m awq.entry --model_path $model \
        --tasks wikitext \
        --w_bit 3 --q_group_size 128 \
        --load_awq /share/awq_cache/$model-w3-g128.pt \
        --q_backend fake --dump_fake ./fake/$model-w3-g128 \

done