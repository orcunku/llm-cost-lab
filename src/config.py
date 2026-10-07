"""Central configuration. Override the model with: export LAB_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = os.environ.get("LAB_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
# One folder per model, so changing LAB_MODEL never reuses another model's ONNX export.
MODELS_DIR = ROOT / "models" / MODEL_ID.replace("/", "--")
FP32_DIR = MODELS_DIR / "fp32"
INT8_DIR = MODELS_DIR / "int8"
INT8_PC_DIR = MODELS_DIR / "int8_pc"   # per-channel INT8, output layer kept in FP32
RESULTS_DIR = ROOT / "results"
BENCH_CSV = RESULTS_DIR / "benchmark.csv"
LOAD_CSV = RESULTS_DIR / "loadtest.csv"
REPORT_MD = RESULTS_DIR / "REPORT.md"
EVAL_TEXTS = ROOT / "data" / "eval_texts.txt"

# One workload for benchmark and load test, so capacity numbers describe the requests that were benchmarked.
PROMPT_LENS = [64, 256]
NEW_TOKENS = 32
LOAD_PROMPT_TOKENS = PROMPT_LENS[len(PROMPT_LENS) // 2]   # the prompt length the report headline uses
# p95 end-to-end latency target for one such request (256 prompt + 32 generated tokens, not streamed).
# Assumption sized to this workload: on a 2-core CPU a single request alone takes ~2.6 s (p50).
DEFAULT_SLA_MS = 4000

# Assumption: price of a 2 vCPU cloud VM (matches a default 2-core Codespace). Replace with your real number.
DEFAULT_HOURLY_USD = 0.085
