from collections import defaultdict
import copy
import json
import os
from pp_utils import get_loaders
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
from torch import nn
import torch
import pdb
import argparse
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    set_seed, 
    Seq2SeqTrainer,
    LlamaTokenizerFast
)
from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftModel
)
import tensorboard
torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

def get_accelerate_model(model_path, ckpt_path=None, lora_path=None, output_path=None):
    print(f'loading base model {model_path}...')
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if ckpt_path:
        print(f'loading ckpt {ckpt_path}...')
        model.load_state_dict(torch.load(ckpt_path), strict=False)
    if lora_path is not None:
        print(f'loading lora adpater {lora_path}...')
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    return model
    
def evaluate(model, tokenizer_path, logger):
    results = {}
    seqlen = 2048
    seed = 42
    if True:
        # for dataset in ["wikitext2", "ptb", "c4","ptb-new",'c4-new']:
        for dataset in ["wikitext2", "c4"]:
            if "llama2" or "Llama" in tokenizer_path:
                cache_testloader = f'./testloader_llama2_{dataset}_all.cache'
            elif "opt" in tokenizer_path:
                cache_testloader = f'./testloader_opt_{dataset}_all.cache'
            else:
                cache_testloader = f'./testloader_llama_{dataset}_all.cache'
            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader)
                logger.info(f"load calibration from {cache_testloader}")
            else:
                dataloader, testloader = get_loaders(
                    dataset,
                    seed=seed,
                    model=tokenizer_path,
                    seqlen=seqlen,
                )
                torch.save(testloader, cache_testloader)
            if "c4" in dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            nsamples = testenc.numel() // seqlen
            use_cache = model.config.use_cache
            model.config.use_cache = False
            model.eval()
            nlls = []
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].cuda()
                # pdb.set_trace()
                # logits = model(batch)['logits']
                outputs = model.model(batch)
                logits = outputs[0]
                logits = model.lm_head(logits)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][
                    :, 1:
                ].to(model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * seqlen
                nlls.append(neg_log_likelihood)

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
            logger.info(f'{dataset} : {ppl.item()}')
            model.config.use_cache = use_cache
            results[dataset] = ppl.item()
            print("dataset:", ppl.item())
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/share/Meta-Llama-3-8B",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="PTQ ckpt",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="lora path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='/share/LLM_EVAL/QuIP/llama3-8b-w4',
        help="output path",
    )
    args = parser.parse_args()
    # model_path = "/mnt/data/share/models/llama-7b"
    # lora_path = "/mnt/data/wangming/root/newBiQuant/qat/outputs/llama7b-mix-4-0.2/20000-128"
    # output_path = "/mnt/data/wangming/root/newBiQuant/merged/llama-7b-mix-4-0.2-20000-128"
    # ckpt = "/mnt/data/wangming/root/newBiQuant/ptq_result/s1/llama7b-mix-4-0.2.pt"
    model = get_accelerate_model(args.model_path, args.ckpt, args.lora_path, args.output_path)
    # model_path = "/mnt/data/share/models/llama-7b"
    # ckpt = "/mnt/data/wangming/root/newBiQuant/ptq_result/lora/llama7b-lora-mix-2-0.1-20000.pt"
    # model = get_accelerate_model(model_path, ckpt)
    # model.eval()
    # import pdb
    # pdb.set_trace()
    # print(model.device)
    # for n,p in model.named_parameters():
    #     p.requires_grad = False
    # results = evaluate(model, args.model_path, logger)
    # print('perplexity result:')
    # for k,v in results.items():
    #     print(k, v)