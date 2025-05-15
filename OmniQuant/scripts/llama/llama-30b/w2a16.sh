CUDA_VISIBLE_DEVICES=7 python main.py \
--model /share/llama-30b --eval_ppl \
--epochs 40 --output_dir ./log/llama-30b-w2a16 \
--wbits 2 --abits 16 --lwc --aug_loss \
--ckpt /share/tmp/llama-30b-omni-w2a16.pt