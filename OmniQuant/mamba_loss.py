import re
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import math

# 1. 解析日志文件
def parse_log(file_path):
    """
    解析日志文件，提取层数、迭代次数和损失值。
    
    Args:
        file_path (str): 日志文件的路径。
    
    Returns:
        pd.DataFrame: 包含 'layer', 'iter', 'loss' 列的数据框。
    """
    # 正则表达式模式，用于提取层数、迭代次数和损失值
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

# 2. 组织数据
log_file_path = '/mnt/data/wangming/OmniQuant/log/GPTQ/mamba-790m_w4/log_rank0_1736231461.txt'  # 替换为你的日志文件路径
df = parse_log(log_file_path)

# 检查数据
print("解析后的数据预览：")
print(df.head())

# 添加 'step' 列，表示每个损失值的顺序
df = df.reset_index().rename(columns={'index': 'step'})
# 从1开始
df['step'] = df['step'] + 1

# 确保数据按出现顺序排序
df.sort_values(by='step', inplace=True)

# 3. 自定义绘图
plt.figure(figsize=(12, 6))
plt.plot(df['step'], df['loss'], label='Loss', color='blue', linewidth=1)

# 设置标题和标签
plt.title('mamba-790m-omniquant-w4')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.yscale('log')
# 计算需要标注的步数（每20步）
step_interval = 20
total_steps = df['step'].max()
num_labels = math.ceil(total_steps / step_interval)

# 生成层标签
layer_labels = []
layer_steps = []
for i in range(num_labels):
    step = i * step_interval
    if step == 0:
        step = 1  # 避免step=0的情况
    layer_label = f'layer{i}'
    layer_labels.append(layer_label)
    layer_steps.append(step)

# 设置 x 轴的刻度和标签
plt.xticks(layer_steps, layer_labels, rotation=45)

# 添加网格
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# 调整布局以防止标签重叠
plt.tight_layout()

# 保存图像
plt.savefig('./mamba790m.png')

# 显示图像（可选）
# plt.show()

# 4. 上传图像到 wandb
# 初始化 wandb
wandb.init(
    project="layer_loss_visualization",  # 替换为你的项目名称
    config={
        "log_file": log_file_path,
        "total_losses": len(df),
    },
    name="mamba-790m-omniquant-w4"  # 可选：为此次运行命名
)

# 上传图像
wandb.log({"All Layers All Iters Loss Plot": wandb.Image('./mamba790m.png')})

# 结束 wandb 运行
wandb.finish()
