import torch
from torch import nn
from tqdm import tqdm
from data_utils import *
@torch.no_grad()
def llm_eval(model, args, tokenizer, dev):
    print('Evaluating ...')
    for dataset in ["wikitext2","c4"]:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen, tokenizer=tokenizer
        )
        if "c4" in dataset:
            testenc = testloader
        else:
            testenc = testloader.input_ids

        nsamples = testenc.numel() // 2048

        use_cache = model.config.use_cache
        model.config.use_cache = False
        layers = model.model.layers

        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        layers[0] = layers[0].to(dev)

        dtype = next(iter(model.parameters())).dtype
        inps = torch.zeros(
            (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
        )
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
        for i in range(nsamples):
            batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
            try:
                model(batch)
            except ValueError:
                pass
        layers[0] = layers[0].module

        layers[0] = layers[0].cpu()
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        torch.cuda.empty_cache()

        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        position_ids = cache['position_ids']

        for i in range(len(layers)):
            layer = layers[i].to(dev)
            for j in range(nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            layers[i] = layer.cpu()
            del layer
            torch.cuda.empty_cache()
            inps, outs = outs, inps

        if model.model.norm is not None:
            model.model.norm = model.model.norm.to(dev)
        model.lm_head = model.lm_head.to(dev)

        testenc = testenc.to(dev)
        nlls = []
        for i in range(nsamples):
            hidden_states = inps[i].unsqueeze(0)
            if model.model.norm is not None:
                hidden_states = model.model.norm(hidden_states)
            lm_logits = model.lm_head(hidden_states)
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = testenc[
                :, (i * model.seqlen):((i + 1) * model.seqlen)
            ][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * model.seqlen
            nlls.append(neg_log_likelihood)
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
        print(ppl.item())

        model.config.use_cache = use_cache
    
@torch.no_grad()
def mamba_eval(model, args, tokenizer, device):
    print('Evaluating ...')

    for dataset in ["wikitext2","c4"]:
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
        '--model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--seqlen',
        type=int, default=2048, help='Sequence length for evaluating PPL'
    )

    args = parser.parse_args()
    if "mamba" in args.model.lower():
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        model = MambaLMHeadModel.from_pretrained(pretrained_model_name=args.model, dtype=torch.bfloat16, device="cuda")
        tokenizer = AutoTokenizer.from_pretrained('/path/to/gpt-neox-20b')
        model.seqlen = args.seqlen
        model.eval()
        mamba_eval(model, args, tokenizer, device=torch.device("cuda:0"))
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype='auto', trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model.seqlen = args.seqlen
        model.eval()
        llm_eval(model, args, tokenizer, dev=torch.device("cuda:0"))

    