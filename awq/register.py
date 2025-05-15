from utils.register import register_method
import subprocess
import os

@register_method("awq")
def run(config):
    model_path = config["model_path"]
    model_name = config["model_name"]
    w_bit = config["w_bit"]
    q_group_size = config["q_group_size"]
    run_awq = config.get("run_awq", False)
    q_backend = config.get("q_backend", "fake")
    tasks = config.get("tasks", "wikitext")

    awq_cache_file = f"{config.get('awq_cache_dir', 'awq_cache')}/{model_name}-w{w_bit}-g{q_group_size}.pt"
    # entry_py = os.path.join(os.path.dirname(__file__), "awq", "entry")
    env = os.environ.copy()

    # Step 1: run AWQ search (optional)
    if run_awq:
        search_cmd = [
            "python", "awq/awq/entry.py",
            "--model_path", model_path,
            "--w_bit", str(w_bit),
            "--q_group_size", str(q_group_size),
            "--run_awq",
            "--dump_awq", awq_cache_file
        ]
        print("Running AWQ search:", " ".join(search_cmd))
        subprocess.run(search_cmd, env=env, check=True)

    # Step 2: evaluate quantized model
    eval_cmd = [
        "python", "awq/awq/entry.py",
        "--model_path", model_path,
        "--tasks", tasks,
        "--w_bit", str(w_bit),
        "--q_group_size", str(q_group_size),
        "--load_awq", awq_cache_file,
        "--q_backend", q_backend
    ]
    print("Running AWQ evaluation:", " ".join(eval_cmd))
    subprocess.run(eval_cmd, env=env, check=True)
