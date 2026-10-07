import os
import subprocess
import sys

from src.config import ROOT


def _fp32_dir(model_id):
    code = "from src.config import FP32_DIR; print(FP32_DIR)"
    env = {**os.environ, "LAB_MODEL": model_id}
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True,
                          check=True).stdout.strip()


def test_each_model_gets_its_own_export_folder():
    a = _fp32_dir("Qwen/Qwen2.5-0.5B-Instruct")
    b = _fp32_dir("HuggingFaceTB/SmolLM2-135M-Instruct")
    assert a != b
    assert "Qwen--Qwen2.5-0.5B-Instruct" in a and "HuggingFaceTB--SmolLM2-135M-Instruct" in b
