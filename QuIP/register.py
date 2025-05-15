from utils.register import register_method
import os
import subprocess
@register_method("quip")
def run(config):
    cmd = [
        "python", "QuIP/run.py",  # 或其他主脚本
        config["model_path"],
        config["dataset"],
        "--wbits", str(config["wbits"]),
        "--quant", config["quant"],
        "--pre_proj_extra", str(config.get("pre_proj_extra", 1)),
        "--qfn", config["qfn"],
        "--save", config["save_path"]
    ]

    # 添加 boolean flags（只加存在且为 True 的）
    for flag in ["pre_gptqH", "pre_rescale", "pre_proj"]:
        if config.get(flag, False):
            cmd.append(f"--{flag}")

    # 设置 CUDA 环境
    env = os.environ.copy()
    if "CUDA_VISIBLE_DEVICES" in config:
        env["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]

    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)
