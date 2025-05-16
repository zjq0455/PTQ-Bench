import re
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import math

def parse_log(file_path):

    pattern = re.compile(
        r'layer\s+(?P<layer>\d+)\s+iter\s+(?P<iter>\d+)\s+loss:(?P<loss>[0-9.e-]+)'
    )
    
    data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                layer = int(match.group('layer'))
                iter_num = int(match.group('iter'))
                loss = float(match.group('loss'))
                data.append({'layer': layer, 'iter': iter_num, 'loss': loss})
    
    df = pd.DataFrame(data)
    return df

log_file_path = '/OmniQuant/log/GPTQ/mamba-790m_w4/log_rank0_1736231461.txt' 
df = parse_log(log_file_path)

print(df.head())

df = df.reset_index().rename(columns={'index': 'step'})

df['step'] = df['step'] + 1

df.sort_values(by='step', inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(df['step'], df['loss'], label='Loss', color='blue', linewidth=1)

plt.title('mamba-790m-omniquant-w4')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.yscale('log')

step_interval = 20
total_steps = df['step'].max()
num_labels = math.ceil(total_steps / step_interval)

layer_labels = []
layer_steps = []
for i in range(num_labels):
    step = i * step_interval
    if step == 0:
        step = 1 
    layer_label = f'layer{i}'
    layer_labels.append(layer_label)
    layer_steps.append(step)

plt.xticks(layer_steps, layer_labels, rotation=45)

plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()

plt.savefig('./mamba790m.png')

wandb.init(
    project="layer_loss_visualization",  
    config={
        "log_file": log_file_path,
        "total_losses": len(df),
    },
    name="mamba-790m-omniquant-w4" 
)

wandb.log({"All Layers All Iters Loss Plot": wandb.Image('./mamba790m.png')})
wandb.finish()
