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

from .config import BENCH_CSV, DEFAULT_HOURLY_USD, MODEL_ID, RESULTS_DIR, ROOT
from .data import load_eval_texts
from .engines import build_engine
from .report import add_cost_columns, baseline_engine
from .stats import summarize


def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return (time.perf_counter() - t0) * 1000, out


def greedy_probes():
    return [t[:200] for t in load_eval_texts()[:5]]


def greedy_path(results_dir, engine):
    return results_dir / f"_greedy_{engine}.json"


def saved_greedy_outputs(results_dir, engine):
    """An engine's saved greedy outputs, or None if missing or produced by another model / probe set."""
    path = greedy_path(results_dir, engine)
    saved = json.loads(path.read_text()) if path.exists() else None
    if isinstance(saved, dict) and saved.get("model") == MODEL_ID and saved.get("probes") == greedy_probes():
        return saved["outputs"]
    return None


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
            if len(out) < 2:
                raise RuntimeError(f"{args.engine} generated {len(out)} token(s); decode speed needs at least 2")
            ttfts.append(ttft)
            e2es.append(e2e)
            tpots.append((e2e - ttft) / (len(out) - 1))   # unclamped: a negative sample is honest timing noise
        t, p, e = summarize(ttfts), summarize(tpots), summarize(e2es)
        if p["p50"] <= 0:
            raise RuntimeError(f"{args.engine}: decode time is lost in timing noise at --new-tokens "
                               f"{args.new_tokens}; use more new tokens or more --runs")
        rows.append({
            "engine": args.engine, "model": MODEL_ID, "target_prompt_tokens": plen, "prompt_tokens": n_in,
            "new_tokens": args.new_tokens, "runs": args.runs,
            "ttft_p50_ms": t["p50"], "ttft_p95_ms": t["p95"], "tpot_p50_ms": p["p50"], "tpot_p95_ms": p["p95"],
            "e2e_p50_ms": e["p50"], "e2e_p95_ms": e["p95"],
            "decode_tps": 1000 / p["p50"], "prefill_tps": n_in / (t["p50"] / 1000),
            "rss_mb": rss, "model_mb": engine.model_mb, "threads": args.threads,
        })

    ppl = engine.perplexity(texts)
    probes = greedy_probes()
    outputs = [engine.generate(p, 24, 24) for p in probes]
    # saved, not compared here: main() compares every engine against whichever baseline the final table has
    greedy_path(results_dir, args.engine).write_text(json.dumps({"model": MODEL_ID, "probes": probes,
                                                                 "outputs": outputs}))
    for r in rows:
        r["perplexity"] = ppl
    pd.DataFrame(rows).to_csv(args.out, index=False)


def previous_rows(csv_path, args):
    """Earlier results, kept only if measured the same way (same model and settings)."""
    if not csv_path.exists():
        return pd.DataFrame()
    old = pd.read_csv(csv_path)
    same = ("model" in old.columns and "target_prompt_tokens" in old.columns and set(old.model) == {MODEL_ID}
            and set(old.new_tokens) == {args.new_tokens} and set(old.target_prompt_tokens) == set(args.prompt_lens))
    if not same:
        print(f"Existing {csv_path.name} used another model or settings; it will be replaced.")
        return pd.DataFrame()
    return old


def greedy_vs_baseline(results_dir, engine, baseline):
    """Recomputed for every engine on every run, so kept rows always use the current baseline."""
    ours, ref = saved_greedy_outputs(results_dir, engine), saved_greedy_outputs(results_dir, baseline)
    if ours is None or ref is None:
        print(f"warning: greedy outputs missing for {engine if ours is None else baseline}; "
              f"greedy_match for {engine} left empty (re-run it)")
        return float("nan")
    return greedy_match(ours, ref)


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
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.new_tokens < 2:
        ap.error("--new-tokens must be at least 2: decode speed is measured from the tokens after the first one")
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if args.worker:
        return run_worker(args)

    csv_path = args.results_dir / "benchmark.csv"
    previous = previous_rows(csv_path, args)
    # earlier engines first: they were saved baseline-first, so a partial re-run keeps the same reference
    baseline = baseline_engine([*(previous.engine.unique() if not previous.empty else []), *args.engines])
    kept = previous[~previous.engine.isin(args.engines)] if not previous.empty else previous
    print(f"Baseline for quality comparison: {baseline}")
    parts = [kept] if not kept.empty else []
    for name in args.engines:
        out = args.results_dir / f"_bench_{name}.csv"
        cmd = [sys.executable, "-m", "src.benchmark", "--worker", "--engine", name, "--out", str(out),
               "--prompt-lens", *map(str, args.prompt_lens), "--new-tokens", str(args.new_tokens),
               "--runs", str(args.runs), "--threads", str(args.threads), "--results-dir", str(args.results_dir)]
        print(f"\n=== benchmarking {name} ===")
        subprocess.run(cmd, check=True, cwd=ROOT)
        parts.append(pd.read_csv(out))
    if not kept.empty:
        print(f"\nKept earlier results for: {', '.join(kept.engine.unique())}")
    df = pd.concat(parts, ignore_index=True).sort_values("engine", key=lambda s: s != baseline, kind="stable")
    matches = {e: greedy_vs_baseline(args.results_dir, e, baseline) for e in df.engine.unique()}
    df["greedy_match"] = df.engine.map(matches)
    df = add_cost_columns(df.reset_index(drop=True), args.hourly_usd)
    df.round(4).to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")
    print(df[["engine", "prompt_tokens", "ttft_p50_ms", "decode_tps", "cost_per_1m_output_tokens_usd",
              "perplexity", "greedy_match"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
