import torch
from mamba_ssm import Mamba
import pdb
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer

import os
import sys
import random
import numpy as np
from datautils import get_loaders
import time
from pprint import pprint
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import utils
# from lm_eval import evaluator

import argparse
from quant import *
from gptq import *

def find_layers(module, layers=[nn.Linear], name=''): #nn.Conv1d, 
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))

    return res

@torch.no_grad()
def main(args):
    # model_name = "/share1/mamba-2.8b"
    print("hello")
    model_name = args.model
    net = model_name.split('/')[-1]

    model = MambaLMHeadModel.from_pretrained(pretrained_model_name=model_name,dtype=torch.bfloat16, device="cuda")
    print("model-ok")
    tokenizer = AutoTokenizer.from_pretrained('/share1/gpt-neox-20b')
    print(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    cache_dataloader = f'./cache/dataloader_{net}_{args.calib_dataset}_calibration.cache'
    if os.path.exists(cache_dataloader):
        dataloader = torch.load(cache_dataloader)
        print(f"load calibration from {cache_dataloader}")
    else:
        dataloader, _ = get_loaders(
            args.calib_dataset,
            tokenizer,
            nsamples=args.nsamples,
            seed=args.seed,
            seqlen=2048,
            model=model_name,
        )
        torch.save(dataloader, cache_dataloader)

    layers = model.backbone.layers

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, 2048, model.config.d_model), dtype=dtype, device=device
    )
    layers[0] = layers[0].to(device)
    cache = {'i': 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, tmp_inf = None, **kwargs):   # forward(self, tmp_cs, tmp_ln, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))   # tmp_num, batch[0].to(device)
        except ValueError:
            pass
    layers[0] = layers[0].module

    # layers[0] = layers[0].cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)

    # ========= start quantization ==========

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(device)
        full = find_layers(layer)
        # full.update({"mixer.A_log":layer.mixer.A_log}) # Not quant?
        # full.update({"mixer.D":layer.mixer.D}) # Not quant?

        sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}
            
            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0))[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print('Quantizing ...')
                gptq[name].fasterquant(args, 
                    percdamp=.01, groupsize=args.groupsize)
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0))[0]

        layers[i] = layer
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps



    # ========= eval ppl ==========
    for dataset in ["wikitext2","ptb","c4"]:
    # for dataset in ["wikitext2", "c4"]:
        cache_testloader = f'./cache/testloader_{net}_{dataset}_all.cache'
        if os.path.exists(cache_testloader):
            testloader = torch.load(cache_testloader)
            print(f"load calibration from {cache_testloader}")
        else:
            _, testloader = get_loaders(
                dataset,
                tokenizer,
                seed=args.seed,
                seqlen=2048,
                model=model_name,
            )
            torch.save(testloader, cache_testloader)
        if "c4" in dataset:
            testenc = testloader
        else:
            testenc = testloader.input_ids

        nsamples = testenc.numel() // 2048
        torch.no_grad()
        model.eval()
        nlls = []
        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * 2048) : ((i + 1) * 2048)].to(device)  #1,2048
            outputs = model(batch) # i, 

            logits = outputs[0]
            shift_logits = logits[:, :-1, :] 
            shift_labels = testenc[:, (i * 2048) : ((i + 1) * 2048)][
                :, 1:
            ].to(device) #1*2047
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * 2048))
        print(f'{dataset} : {ppl.item()}')


if __name__ == "__main__":
    print("Let's start mamba!")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--calib_dataset",type=str,default="wikitext2",
        choices=["wikitext2", "ptb", "c4"], #, "mix","pile"
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        '--nsamples', type=int, default=128, help='Number of calibration data samples.')
    parser.add_argument(
        '--act_quant', action='store_true', help='Whether to use the new PTB and C4 eval.')
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--method', type=str, default="gptq",
        help='Select the quantization method.'
    )


    args = parser.parse_args()

    main(args)