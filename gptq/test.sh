# CUDA_VISIBLE_DEVICES=1 python run.py /share/llama-7b c4 --wbits 2 --save /share/LLM_EVAL/GPTQ/llama-7b-2
# CUDA_VISIBLE_DEVICES=1 python run.py /share/llava-v1.5-7b c4 --wbits 2 --save /share/LLM_EVAL/GPTQ/llava-v1.5-7b-w2 --groupsize 128
# CUDA_VISIBLE_DEVICES=1 python run.py /share/VILA1.5-7b/llm c4 --wbits 2 --save /share/LLM_EVAL/GPTQ/VILA1.5-7b-w2/llm --groupsize 128 --act-order
# CUDA_VISIBLE_DEVICES=1 python run.py /share/deepseek-moe-16b-base c4 --wbits 2 --save /share/LLM_EVAL/GPTQ/deepseek-moe-16b-base-w3 --groupsize 128
# CUDA_VISIBLE_DEVICES=1 python run.py /share/Mixtral-8x7B-v0.1 c4 --wbits 2 --save /share/LLM_EVAL/GPTQ/deepseek-moe-16b-base-w3 --groupsize 128
# CUDA_VISIBLE_DEVICES=1 python run.py /share/Mistral-7B-v0.1 c4 --wbits 2 --save /share/LLM_EVAL/GPTQ/mistral-7b-2 --groupsize 128
CUDA_VISIBLE_DEVICES=1 python run.py /share/mamba-790m c4 --wbits 2