import argparse
import time
import numpy as np
import torch
import torch.nn as nn
# import quant

from gptq import GPTQ
from texttable import Texttable

# added for integration with QuIP
from datautils import get_loaders
from modelutils import * # includes DEV
from quant import Quantizer
from bal import Balance
from near import Nearest

from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Added arguments for integration with QuIP
    parser.add_argument('--quant',
                        choices=['bitbal', 'parbal', 'allbal', 'allbal_block', 'allbal_clipevery', 'allbal_stochinit', 
                        'ldlq', 'ldlqRG', 'ldlqRG_block', 'ldlbal_admm', 'nearest', 'gptq', 'gptq_updown'],
                        default='nearest',
                        help='Which quantization method to use.')
    parser.add_argument('--pre_gptqH', action='store_true',help='preprocessing')
    parser.add_argument('--pre_rescale', action='store_true', help='preprocessing')
    parser.add_argument('--pre_proj', action='store_true', help='preprocessing')
    parser.add_argument( '--pre_proj_extra', type=int, default=0, choices=[0, 1, 2], help='Extra options to control pre_proj step.')
    parser.add_argument('--qfn', type=str, default='a', help='qfn: a is default, b is sym incoherent based')
    parser.add_argument('--npasses', type=int, default=1, help='number passes to repeat balance loop over 1-d.')

    parser.add_argument('model', type=str, help='llama model to load')
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'ptb', 'c4'], help='Where to extract calibration data from.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration data samples.')
    parser.add_argument('--percdamp', type=float, default=.01, help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16], help='#bits to use for quantization; use 16 for evaluating base model.')
    parser.add_argument('--groupsize', type=int, default=-1, help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument('--eval', action='store_true', help='evaluate quantized model.')
    parser.add_argument('--test-generation', action='store_true', help='test generation.')
    parser.add_argument('--save', type=str, default='', help='Save quantized checkpoint under this name.')
    parser.add_argument('--load', type=str, default='', help='Load quantized model.')
    parser.add_argument('--benchmark', type=int, default=0, help='Number of tokens to use for benchmarking.')
    parser.add_argument('--check', action='store_true', help='Whether to compute perplexity during benchmarking for verification.')
    parser.add_argument('--new-eval', action='store_true', help='Whether to use the new PTB and C4 eval')
    parser.add_argument('--layers-dist', type=str, default='', help='Distribution of layers across GPUs. e.g. 2:1:1 for 2 layers on GPU 0, 1 layer on GPU 1, and 1 layer on GPU 2. Any remaining layers will be assigned to your last GPU.')
    parser.add_argument(
        '--unbiased',
        action='store_true',
        help='unbiased')
    args = parser.parse_args()
    tokenizer = None
    if "llama" in args.model.lower():
        from llama import get_llama, llama_sequential
        model = get_llama(args.model)
        sequential = llama_sequential
    elif "llava" in args.model.lower():
        from llava_quip import get_llava, llava_sequential
        model, tokenizer = get_llava(args.model)
        sequential = llava_sequential
    elif "vila" in args.model.lower():
        from llava_quip import get_vila, llava_sequential
        model, tokenizer = get_vila(args.model)
        sequential = llava_sequential
    elif "mamba" in args.model.lower():
        from mamba_quip import get_mamba, mamba_sequential
        model, tokenizer = get_mamba(args.model)
        sequential = mamba_sequential
    elif "mixtral" in args.model.lower() or "deepseek" in args.model.lower() or "mistral" in args.model.lower():
        from mixtral import get_mixtral, mixtral_sequential
        model = get_mixtral(args.model)
        sequential = mixtral_sequential
            
    model.eval()

    dataloader, testloader = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen, tokenizer=tokenizer)

    if not args.load and args.wbits < 16:
        tick = time.time()
        quantizers = sequential(model, dataloader, DEV, args)
        print(time.time() - tick)

    if args.save:
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.save_pretrained(args.save)
        model.save_pretrained(args.save)
        
