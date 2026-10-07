"""Benchmark engines on what LLM serving teams care about:
TTFT, inter-token latency, prefill/decode throughput, memory, and output quality.
Each engine runs in its own subprocess so memory numbers are clean."""
import argparse
import json
import os
import subprocess
import sys

import pandas as pd
import psutil
import time

from .config import BENCH_CSV, DEFAULT_HOURLY_USD, RESULTS_DIR, ROOT
from .data import load_eval_texts
from .engines import build_engine
from .report import add_cost_columns
from .stats import summarize


def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return (time.perf_counter() - t0) * 1000, out


def greedy_match(a, b):
    """Average positional agreement between two lists of token-id lists."""
    scores = []
    for x, y in zip(a, b):
        n = min(len(x), len(y))
        scores.append(sum(1 for i in range(n) if x[i] == y[i]) / max(n, 1))
    return sum(scores) / len(scores) if scores else float("nan")


def run_worker(args):
    results_dir = args.results_dir
    before = rss_mb()
    engine = build_engine(args.engine, args.threads)
    texts = load_eval_texts()
    engine.generate(engine.make_prompt(args.prompt_lens[0]), 4, 4)           # warm-up
    rss = rss_mb() - before

    rows = []
    for plen in args.prompt_lens:
        prompt = engine.make_prompt(plen)
        n_in = engine.count_tokens(prompt)
        ttfts, tpots, e2es = [], [], []
        for _ in range(args.runs):
            ttft, _ = timed(lambda: engine.generate(prompt, 1, 1))
            e2e, out = timed(lambda: engine.generate(prompt, args.new_tokens, args.new_tokens))
            ttfts.append(ttft)
            e2es.append(e2e)
            tpots.append(max(e2e - ttft, 1e-6) / max(len(out) - 1, 1))
        t, p, e = summarize(ttfts), summarize(tpots), summarize(e2es)
        rows.append({
            "engine": args.engine, "prompt_tokens": n_in, "new_tokens": args.new_tokens, "runs": args.runs,
            "ttft_p50_ms": t["p50"], "ttft_p95_ms": t["p95"], "tpot_p50_ms": p["p50"], "tpot_p95_ms": p["p95"],
            "e2e_p50_ms": e["p50"], "e2e_p95_ms": e["p95"],
            "decode_tps": 1000 / p["p50"], "prefill_tps": n_in / (t["p50"] / 1000),
            "rss_mb": rss, "model_mb": engine.model_mb, "threads": args.threads,
        })

    ppl = engine.perplexity(texts)
    probes = [t[:200] for t in texts[:5]]
    outputs = [engine.generate(p, 24, 24) for p in probes]
    greedy_path = results_dir / "_greedy_baseline.json"
    if args.is_baseline:
        greedy_path.write_text(json.dumps(outputs))
        match = 1.0
    elif greedy_path.exists():
        match = greedy_match(outputs, json.loads(greedy_path.read_text()))
    else:
        match = float("nan")
    for r in rows:
        r["perplexity"], r["greedy_match"] = ppl, match
    pd.DataFrame(rows).to_csv(args.out, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+", default=["pytorch", "onnx_fp32", "onnx_int8", "onnx_int8_pc"])
    ap.add_argument("--prompt-lens", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--hourly-usd", type=float, default=DEFAULT_HOURLY_USD)
    ap.add_argument("--results-dir", type=lambda s: __import__("pathlib").Path(s), default=RESULTS_DIR)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--engine")
    ap.add_argument("--is-baseline", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if args.worker:
        return run_worker(args)

    parts = []
    for i, name in enumerate(args.engines):
        out = args.results_dir / f"_bench_{name}.csv"
        cmd = [sys.executable, "-m", "src.benchmark", "--worker", "--engine", name, "--out", str(out),
               "--prompt-lens", *map(str, args.prompt_lens), "--new-tokens", str(args.new_tokens),
               "--runs", str(args.runs), "--threads", str(args.threads), "--results-dir", str(args.results_dir)]
        if i == 0:
            cmd.append("--is-baseline")
        print(f"\n=== benchmarking {name} ===")
        subprocess.run(cmd, check=True, cwd=ROOT)
        parts.append(pd.read_csv(out))
    df = add_cost_columns(pd.concat(parts, ignore_index=True), args.hourly_usd)
    csv_path = args.results_dir / "benchmark.csv"
    df.round(4).to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")
    print(df[["engine", "prompt_tokens", "ttft_p50_ms", "decode_tps", "cost_per_1m_output_tokens_usd",
              "perplexity", "greedy_match"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
