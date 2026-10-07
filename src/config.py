"""Central configuration. Override the model with: export LAB_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = os.environ.get("LAB_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MODELS_DIR = ROOT / "models"
FP32_DIR = MODELS_DIR / "fp32"
INT8_DIR = MODELS_DIR / "int8"
INT8_PC_DIR = MODELS_DIR / "int8_pc"   # per-channel INT8, output layer kept in FP32
RESULTS_DIR = ROOT / "results"
BENCH_CSV = RESULTS_DIR / "benchmark.csv"
LOAD_CSV = RESULTS_DIR / "loadtest.csv"
REPORT_MD = RESULTS_DIR / "REPORT.md"
EVAL_TEXTS = ROOT / "data" / "eval_texts.txt"

# Assumption: price of a 2 vCPU cloud VM (matches a default 2-core Codespace). Replace with your real number.
DEFAULT_HOURLY_USD = 0.085
