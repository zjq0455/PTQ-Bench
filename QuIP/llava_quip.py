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


def get_llava(model):

    def skip(*args, **kwargs):
        pass
    import sys
    sys.path.append("/path/to/LLaVA")
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    enc, model, image_processor, context_len = load_pretrained_model(
    model_path=model,
    model_base=None,
    model_name=get_model_name_from_path(model),
    device="cpu",
    **{"use_cache": False}
    )
    model.seqlen = 2048
    #model.seqlen = 4096
    return model, enc

def get_vila(model_path):
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # Note (Haotian): To avoid OOM after huggingface transformers 4.36.2
    config.use_cache = False
    enc = AutoTokenizer.from_pretrained(
    model_path, use_fast=False, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=True, torch_dtype=torch.float16
    )
    model.seqlen = 2048
    return model, enc

@torch.no_grad()
def llava_sequential(model, dataloader, dev, args):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')
    print(dev)
    quantizers = {}
    for i in tqdm(range(len(layers))):

        layer = layers[i].to(dev)
        subset = find_layers(layer)
        # Initialize Quant Method and Çompute H
        quant_method = {}
        for name in subset:
            if args.quant == 'gptq':
                quant_method[name] = GPTQ(subset[name])
                quant_method[name].quantizer = Quantizer()
                quant_method[name].quantizer.configure(args.wbits,
                                               perchannel=True,
                                               sym=False,
                                               qfn=args.qfn,
                                               mse=False)
            elif args.quant == 'nearest':
                quant_method[name] = Nearest(subset[name])
                quant_method[name].quantizer = Quantizer()
                quant_method[name].quantizer.configure(args.wbits,
                                               perchannel=True,
                                               sym=False,
                                               qfn=args.qfn,
                                               mse=False)
            elif args.quant in ['bitbal','parbal','allbal','allbal_block','allbal_clipevery','allbal_stochinit',
            'ldlq','ldlqRG','ldlqRG_block','ldlbal_admm']:
                quant_method[name] = Balance(subset[name])
                quant_method[name].configure(
                                    args.quant,
                                    args.wbits, 
                                    args.npasses,
                                    unbiased=args.unbiased)
                quant_method[name].quantizer = Quantizer()
                quant_method[name].quantizer.configure(args.wbits,
                                               perchannel=True,
                                               sym=False,
                                               qfn=args.qfn,
                                               mse=False)

        def add_batch(name):

            def tmp(_, inp, out):
                quant_method[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
        # added for QuIP integration
        for name in subset:
            quant_method[name].post_batch()

        for name in subset:
            quant_method[name].preproc(
                                preproc_gptqH=args.pre_gptqH, percdamp=args.percdamp,
                                preproc_rescale=args.pre_rescale, 
                                preproc_proj=args.pre_proj, preproc_proj_extra=args.pre_proj_extra)
            if args.quant == 'gptq':
                quant_method[name].fasterquant(groupsize=args.groupsize)
            elif args.quant in ['bitbal','parbal','allbal','allbal_block','allbal_clipevery','allbal_stochinit',
            'ldlq','ldlqRG','ldlqRG_block','ldlbal_admm']:
                quant_method[name].fasterquant()
            elif args.quant == 'nearest':
                quant_method[name].fasterquant()

            quantizers['model.decoder.layers.%d.%s' %
                    (i, name)] = quant_method[name].quantizer
            quant_method[name].free()


        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del quant_method
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers

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

    if args.layers_dist:
        gpu_dist = [int(x) for x in args.layers_dist.split(':')]
    else:
        gpu_dist = []

    if type(args.load) is not str:
        args.load = args.load.as_posix()

    if args.load:
        model = load_quant(args.model, args.load, args.wbits, args.groupsize)
    else:
        model, tokenizer = get_vila(args.model)
        # model, tokenizer = get_llava(args.model)
        model.eval()

    dataloader, testloader = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen, tokenizer=tokenizer)

    if not args.load and args.wbits < 16:
        tick = time.time()
        quantizers = llava_sequential(model, dataloader, DEV, args)
        print(time.time() - tick)


    if args.save:
        tokenizer.save_pretrained(args.save)
        model.save_pretrained(args.save)
        
    if args.eval:
        datasets = ['wikitext2', 'ptb', 'c4']
        if args.new_eval:
            datasets = ['wikitext2', 'ptb-new', 'c4-new']
            datasets = ['c4-new']
        for dataset in datasets:
            dataloader, testloader = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)
            print(dataset)
            llama_eval(model, testloader, DEV)
    
    if args.test_generation:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            llama_multigpu(model, gpus, gpu_dist)
        else:
            model = model.to(DEV)

        from transformers import LlamaTokenizer, TextStreamer
        tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)
        input_ids = tokenizer(["The capital of New Mexico is"], return_tensors="pt").input_ids.to(gpus[0])
        streamer = TextStreamer(tokenizer)
        with torch.no_grad():
            generated_ids = model.generate(input_ids, streamer=streamer)
