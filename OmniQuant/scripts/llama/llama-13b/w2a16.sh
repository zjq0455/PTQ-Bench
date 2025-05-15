CUDA_VISIBLE_DEVICES=2 python main.py \
--model /share/llama-13b --eval_ppl \
--epochs 40 --output_dir ./log/llama-13b-w2a16 \
--wbits 2 --abits 16 --lwc --aug_loss