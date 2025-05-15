import torch
import sys
# sys.path.append("/mnt/data/wangming/LLaVA")
# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
# from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
# def get_mamba(model):
#     model = MambaLMHeadModel.from_pretrained(pretrained_model_name=model,dtype=torch.bfloat16, device="cuda")
#     tokenizer = AutoTokenizer.from_pretrained('/share/gpt-neox-20b')
#     model.seqlen = 2048
#     return model, tokenizer
model_path = "/share/VILA1.5-7b/llm"
ckpt_path = "/share/LLM_EVAL/OmniQuant/VILA1.5-7b-w4.pt"
output_path = "/share/LLM_EVAL/OmniQuant/VILA1.5-7b-w4/llm"
# model, enc = get_mamba(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype='auto', trust_remote_code=True)
# enc = AutoTokenizer.from_pretrained(model_path)



config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
# # Note (Haotian): To avoid OOM after huggingface transformers 4.36.2
kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
enc = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )
model = AutoModelForCausalLM.from_pretrained(
    model_path, config=config, trust_remote_code=True, **kwargs
)


# enc, model, image_processor, context_len = load_pretrained_model(
# model_path=model_path,
# model_base=None,
# model_name=get_model_name_from_path(model_path),
# device="cpu",
# **{"use_cache": False}
# )

# enc = AutoTokenizer.from_pretrained(
#         model_path, use_fast=False, trust_remote_code=True
#     )
# model = AutoModelForCausalLM.from_pretrained(
# model_path, device_map="auto", torch_dtype=torch.float16
# )
model.eval()
print(f'loading base model {model_path}...')
if ckpt_path:
    print(f'loading ckpt {ckpt_path}...')
    model.load_state_dict(torch.load(ckpt_path), strict=False)
model.save_pretrained(output_path)
enc.save_pretrained(output_path)

