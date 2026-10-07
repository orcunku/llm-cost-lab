"""Closed-loop HTTP load test: latency, throughput and queueing delay vs concurrency."""
import argparse
import asyncio
import subprocess
import sys
import time

import httpx
import pandas as pd

from .config import LOAD_CSV, LOAD_PROMPT_TOKENS, NEW_TOKENS, RESULTS_DIR, ROOT
from .stats import summarize


async def run_level(client, concurrency, duration_s, prompt, max_new_tokens):
    latencies, queues, computes = [], [], []
    tokens, errors = 0, 0
    t_start = time.perf_counter()
    deadline = t_start + duration_s

    async def worker():
        nonlocal tokens, errors
        while time.perf_counter() < deadline:
            t0 = time.perf_counter()
            try:
                r = await client.post("/generate", json={"prompt": prompt, "max_new_tokens": max_new_tokens})
                r.raise_for_status()
                d = r.json()
                latencies.append((time.perf_counter() - t0) * 1000)
                queues.append(d["queue_ms"])
                computes.append(d["compute_ms"])
                tokens += d["new_tokens"]
            except Exception:
                errors += 1

    await asyncio.gather(*[worker() for _ in range(concurrency)])
    elapsed = time.perf_counter() - t_start
    row = {"concurrency": concurrency, "requests": len(latencies), "errors": errors}
    if latencies:
        s = summarize(latencies)
        row.update({"rps": len(latencies) / elapsed, "tokens_per_sec": tokens / elapsed,
                    "p50_ms": s["p50"], "p95_ms": s["p95"], "p99_ms": s["p99"],
                    "mean_queue_ms": sum(queues) / len(queues), "mean_compute_ms": sum(computes) / len(computes)})
    return row


async def run_all(base_url, levels, duration_s, prompt, max_new_tokens):
    rows = []
    async with httpx.AsyncClient(base_url=base_url, timeout=600) as client:
        await client.post("/generate", json={"prompt": prompt, "max_new_tokens": 2})   # warm-up
        for c in levels:
            row = await run_level(client, c, duration_s, prompt, max_new_tokens)
            rows.append(row)
            print(f"  concurrency={c:<3} rps={row.get('rps', 0):.2f} p95={row.get('p95_ms', float('nan')):.0f}ms "
                  f"queue={row.get('mean_queue_ms', float('nan')):.0f}ms errors={row['errors']}")
    return rows


def wait_for_server(url, proc, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError("server process exited early; check the error above")
        try:
            if httpx.get(url + "/health", timeout=2).status_code == 200:
                return
        except Exception:
            time.sleep(1)
    raise TimeoutError("server did not start in time")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="onnx_int8")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--duration", type=float, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=NEW_TOKENS)
    ap.add_argument("--prompt-tokens", type=int, default=LOAD_PROMPT_TOKENS)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--url", default=None, help="use an already running server instead of starting one")
    ap.add_argument("--out", default=str(LOAD_CSV))
    args = ap.parse_args()

    url = args.url or f"http://127.0.0.1:{args.port}"
    proc = None
    if not args.url:
        proc = subprocess.Popen([sys.executable, "-m", "src.server", "--engine", args.engine,
                                 "--port", str(args.port)], cwd=ROOT)
    try:
        wait_for_server(url, proc)
        model = httpx.get(url + "/health", timeout=10).json().get("model", "unknown")
        p = httpx.get(url + "/prompt", params={"tokens": args.prompt_tokens}, timeout=60).json()
        print(f"Load testing engine={args.engine} model={model} "
              f"prompt_tokens={p['prompt_tokens']} new_tokens={args.max_new_tokens}")
        rows = asyncio.run(run_all(url, args.levels, args.duration, p["prompt"], args.max_new_tokens))
    finally:
        if proc is not None:
            proc.terminate()
    RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.insert(0, "engine", args.engine)
    df.insert(1, "model", model)
    df.insert(2, "prompt_tokens", p["prompt_tokens"])
    df.insert(3, "new_tokens", args.max_new_tokens)
    df.round(3).to_csv(args.out, index=False)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
