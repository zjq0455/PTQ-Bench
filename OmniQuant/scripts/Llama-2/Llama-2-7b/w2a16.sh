CUDA_VISIBLE_DEVICES=0 python main.py \
--model /share/Llama-2-7b-hf --eval_ppl \
--epochs 40 --output_dir ./log/Llama-2-7b-w2a16 \
--wbits 2 --abits 16 --lwc --aug_loss