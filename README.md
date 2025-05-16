# 🔬 PTQ-Bench

This repository contains the evaluation codes for the paper **[Benchmarking Post-Training Quantization in LLMs: Comprehensive Taxonomy, Unified Evaluation, and Comparative Analysis]**. Each method (**GPTQ**, **AWQ**, **OmniQuant**, and **QuIP**) is modularized, configurable via YAML, and supports streamlined evaluation via a common launcher.

---

## 🚀 Usage

### 1. Environment Setup

```bash
conda create -n quant-bench python=3.10
conda activate quant-bench
pip install -r requirements.txt
```

> You should install the Mamba and AWQ environments separately by following their official repositories.

---

### 2. Run Quantization

Use the launcher `run_quant.py` with `--method` and `--config`:

```bash
python run_quant.py --method gptq --config configs/gptq.yaml
python run_quant.py --method omniquant --config configs/omniquant.yaml
python run_quant.py --method quip --config configs/quip.yaml
python run_quant.py --method awq --config configs/awq.yaml
```

---

### 3. Example Config: `configs/gptq.yaml`

```yaml
model_path: /PATH/TO/llama-7b
dataset: c4
wbits: 2
save_path: /PATH/TO/GPTQ/llama-7b-w2
act_order: true
CUDA_VISIBLE_DEVICES: "1"
```

---

## 4. Perplexity Evaluation

1. Save the quantized model weights.
2. Run the following command in your terminal:

```bash
python eval_ppl.py --model /PATH/TO/GPTQ/llama-7b-w2
```

---

## 5. Evaluation of Zero-shot Tasks

We use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to evaluate zero-shot performance. To run an evaluation, you can use a script like the following:

```bash
TASKS="truthfulqa,hellaswag,winogrande,race,piqa,mmlu,hellaswag,arc_easy,arc_challenge,lambada,gsm8k,ceval-valid"
CUDA_VISIBLE_DEVICES=5 lm_eval --model hf \
        --model_args pretrained=/PATH/TO/GPTQ/llama-7b-w2 \
        --tasks $TASKS \
        --device cuda:0 \
        --batch_size auto:4 \
        --output ./results/GPTQ/llama-7b-w2
```

## 6. Evaluation of Multi-Modal Tasks

We use the official repository to evaluate LLaVA and VILA. For details, refer to the [Evaluation Guide](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).

## Related Projects

[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers][https://github.com/IST-DASLab/gptq]

[OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models][https://github.com/OpenGVLab/OmniQuant]

[QuIP: 2-Bit Quantization of Large Language Models With Guarantees][https://github.com/Cornell-RelaxML/QuIP]

[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration][https://github.com/mit-han-lab/llm-awq]

[Visual Instruction Tuning][https://github.com/haotian-liu/LLaVA]

[VILA: On Pre-training for Visual Language Models][https://github.com/NVlabs/VILA]

[Mamba: Linear-Time Sequence Modeling with Selective State Spaces][https://github.com/state-spaces/mamba]

[lm-evaluation-harness][https://github.com/EleutherAI/lm-evaluation-harness] 