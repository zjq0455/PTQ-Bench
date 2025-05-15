import time

import torch
import torch.nn as nn

from gptq import *
from modelutils import *
from quant import *


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
        '--save', type=str, default=None,
        help='Where to save quantized model'
    )

    args = parser.parse_args()
    tokenizer = None
    if "llama" in args.model.lower():
        from llama import get_llama, llama_sequential
        model = get_llama(args.model)
        sequential = llama_sequential
    elif "llava" in args.model.lower() or "vila" in args.model.lower():
        from llava_gptq import get_llava, llava_sequential
        model, tokenizer = get_llava(args.model)
        sequential = llava_sequential
    elif "deepseek" in args.model.lower() or "mixtral" in args.model.lower() or "mistral" in args.model.lower():
        from mistral import get_mixtral, mixtral_sequential
        model = get_mixtral(args.model)
        sequential = mixtral_sequential
    elif "mamba" in args.model.lower():
        from mamba_gptq import get_mamba,  mamba_sequential
        model, tokenizer = get_mamba(args.model)
        sequential = mamba_sequential
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
        quantizers = sequential(model, dataloader, DEV, args)
        print(time.time() - tick)

    if args.save:
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.save_pretrained(args.save)
        model.save_pretrained(args.save)
