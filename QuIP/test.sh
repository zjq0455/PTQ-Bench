CUDA_VISIBLE_DEVICES=1 python run.py /share/llama-7b c4 --wbits 2 --quant ldlq --pre_gptqH --pre_rescale --pre_proj --pre_proj_extra 1 --qfn 'b' --eval
CUDA_VISIBLE_DEVICES=1 python run.py /share/llava-v1.5-7b c4 --wbits 3 --quant ldlq --pre_gptqH --pre_rescale --pre_proj --pre_proj_extra 1 --qfn 'b'
CUDA_VISIBLE_DEVICES=1 python run.py /share/VILA1.5-7b/llm c4 --wbits 3 --quant ldlq --pre_gptqH --pre_rescale --pre_proj --pre_proj_extra 1 --qfn 'b'
CUDA_VISIBLE_DEVICES=1 python run.py /share/Mixtral-8x7B-v0.1 c4 --wbits 4 --quant ldlq --pre_gptqH --pre_rescale --pre_proj --pre_proj_extra 1 --qfn 'b'
CUDA_VISIBLE_DEVICES=1 python run.py /share/deepseek-moe-16b-base c4 --wbits 2 --quant ldlq --pre_gptqH --pre_rescale --pre_proj --pre_proj_extra 1 --qfn 'b'
CUDA_VISIBLE_DEVICES=1 python run.py /share/Mistral-7B-v0.1 c4 --wbits 2 --quant ldlq --pre_gptqH --pre_rescale --pre_proj --pre_proj_extra 1 --qfn 'b'
CUDA_VISIBLE_DEVICES=1 python run.py /share/mamba-2.8b c4 --wbits 2 --quant ldlq --pre_gptqH --pre_rescale --pre_proj --pre_proj_extra 1 --qfn 'b'