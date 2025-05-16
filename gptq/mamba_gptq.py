import time

import torch
import torch.nn as nn
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from gptq import *
from modelutils import *
from quant import *
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def get_mamba(model):
    model = MambaLMHeadModel.from_pretrained(pretrained_model_name=model,dtype=torch.bfloat16, device="cuda")
    tokenizer = AutoTokenizer.from_pretrained('/path/to/gpt-neox-20b')
    model.seqlen = 2048
    return model, tokenizer

@torch.no_grad()
def mamba_sequential(model, dataloader, dev, args):
    print('Starting ...')
    layers = model.backbone.layers


    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, 2048, model.config.d_model), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, tmp_inf = None, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module


    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)
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
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0))[0]

        layers[i] = layer
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps
    
    return quantizers

@torch.no_grad()
def mamba_eval(model, args, tokenizer, device):
    import os
    print('Evaluating ...')

    for dataset in ["wikitext2","c4"]:
    # for dataset in ["wikitext2", "c4"]:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen, tokenizer=tokenizer
        )
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


if __name__ == '__main__':
    import argparse
    from data_utils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )
    parser.add_argument(
        '--ckpt', type=str,
        help='Whether to save quantized model'
    )

    args = parser.parse_args()

    model, tokenizer = get_mamba(args.model)
    model.eval()
    torch.backends.cudnn.benchmark = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen, tokenizer=tokenizer
    )

    if args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = mamba_sequential(model, dataloader, DEV, args)
        print(time.time() - tick)

    mamba_eval(model, args, tokenizer, device=DEV)