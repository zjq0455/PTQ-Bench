from utils.register import register_method
import os
import subprocess

@register_method("omniquant")
def run(config):
    cmd = [
        "python", "OmniQuant/main.py",  # 改为你主脚本相对路径
        "--model", config["model"],
        "--epochs", str(config["epochs"]),
        "--output_dir", config["output_dir"],
        "--wbits", str(config["wbits"]),
        "--abits", str(config["abits"]),
        "--nsamples", str(config["nsamples"]),
        "--ckpt", config["ckpt"]
    ]

    # 加入 boolean flags（只加 true 的）
    for flag in ["lwc", "eval_ppl"]:
        if config.get(flag, False):
            cmd.append(f"--{flag}")

    env = os.environ.copy()
    if "cuda_visible_devices" in config:
        env["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)
