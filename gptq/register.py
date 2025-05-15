from utils.register import register_method
import os
import subprocess
@register_method("gptq")
def run(config):
    model_path = config["model_path"]
    dataset = config["dataset"]
    wbits = str(config["wbits"])
    save_path = config["save_path"]
    device = config.get("CUDA_VISIBLE_DEVICES", None)
    act_order = config.get("act_order", False)
    cmd = [
        "python", "gptq/run.py",
        model_path, dataset,
        "--wbits", wbits,
        "--save", save_path
    ]
    if act_order:
        cmd.append("--act-order")
    if device:
        env = {"CUDA_VISIBLE_DEVICES": device, **os.environ}
    else:
        env = os.environ
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)