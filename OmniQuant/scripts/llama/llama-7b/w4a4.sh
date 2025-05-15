CUDA_VISIBLE_DEVICES=6 python main.py \
--model /mnt/data/share/models/llama-7b --eval_ppl \
--epochs 20 --output_dir ./log/llama-7b-w4a4 \
--wbits 4 --abits 4 --lwc --let --aug_loss --mean_loss \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande