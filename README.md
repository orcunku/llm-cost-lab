# LLM Inference Cost Lab

**Live demo:** [llm-cost-lab.streamlit.app](https://llm-cost-lab.streamlit.app): explore the measured results,
load-test curves and the capacity & cost planner in the browser, with no install.

**Question answered:** *How much cheaper is it to serve a small LLM with ONNX Runtime (FP32 / INT8) than with PyTorch,
what does quantization cost in quality, and how many instances (and dollars per month) do I need for a target
traffic level and latency SLA?*

Everything runs on CPU with free tools (GitHub Codespaces + open models).

```
 export (ONNX + KV-cache) -> INT8 quantization
          |
          v
 benchmark  (per engine, isolated process)      loadtest (real HTTP server, concurrency sweep)
  TTFT, inter-token latency, tokens/s,           p50/p95/p99, req/s, queue vs compute time
  memory, perplexity, output agreement                       |
          \_____________________  ___________________________/
                                 \/
              report.py -> results/REPORT.md  +  Streamlit dashboard
              (cost per 1M input/output tokens, instances needed, monthly cost, self-host vs API break-even)
```

## Run it (GitHub Codespaces)
```bash
bash setup.sh        # automatic on first start (3-5 min)
bash run_all.sh      # export -> benchmark -> load test -> report -> dashboard (about 20-40 min on 2 cores)
```
Each step can be run alone:
```bash
python -m src.export
python -m src.benchmark --runs 10 --prompt-lens 64 256 --new-tokens 32
python -m src.loadtest --engine onnx_int8_mixed --levels 1 2 4 8 --duration 40
python -m src.report --hourly-usd 0.085 --sla-ms 4000 --target-rps 5
pytest -q            # works without torch: tests use a simulated engine
```
Faster first try: `export LAB_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct` before step 1 (set it in the same terminal).
Each model gets its own export folder under `models/`, and every result row records the model it measured.
Re-running only some engines keeps earlier results as long as model and settings match.

## Metrics
| Metric | Why it matters |
|---|---|
| TTFT (time to first token) | what a user feels before text starts appearing |
| Inter-token latency, decode tokens/s | streaming speed and output-token cost |
| Prefill tokens/s | input-token cost (long prompts, RAG) |
| Cost per 1M input / output tokens | same unit hosted APIs bill in |
| Perplexity + greedy agreement | proves the cheaper model is still acceptable |
| p95 vs concurrency, queue vs compute | capacity planning against an SLA |
| Instances + monthly cost, API break-even | the decision a manager actually needs |

## Results
SmolLM2-135M-Instruct on a 2-vCPU Codespace, 256 prompt tokens + 32 generated tokens per request
(full report: [`results/smollm2-135m/REPORT.md`](results/smollm2-135m/REPORT.md)).

| Engine | TTFT p50 | Decode tok/s | $/1M output tok | Perplexity | Memory (RSS) | Quality gate |
|---|---|---|---|---|---|---|
| pytorch (baseline) | 521 ms | 15.0 | $1.57 | 33.49 | 857 MB | baseline |
| onnx_fp32 | 631 ms | 17.5 | $1.35 | 33.49 (+0%) | 1358 MB | pass |
| onnx_int8 | 300 ms | 36.6 | $0.64 | 55.66 (+66%) | 700 MB | **fail** |
| onnx_int8_pc | 393 ms | 29.0 | $0.81 | 52.90 (+58%) | 775 MB | **fail** |
| **onnx_int8_mixed** | 462 ms | **24.1** | **$0.98** | 34.31 (+2.4%) | 946 MB | pass |

**Recommendation:** deploy `onnx_int8_mixed`: INT8 everywhere except the MLP `down_proj` layers, the output
layer and the embedding table. It decodes 1.61x faster and costs 38% less per output token than PyTorch, with
+2.4% perplexity (gate: 5%). One 2-vCPU instance serves 0.57 req/s within a 4 s p95 SLA, so 5 req/s needs
11 instances, about $683/month at $0.085/hour (17 instances and ~$1,055 with ONNX FP32).

**What the numbers show**
- **Where INT8 breaks matters more than whether to use it.** Plain INT8 is 2.4x faster but raises perplexity
  by 66%. Quantizing one layer group at a time showed that almost all of the damage comes from the 30 MLP
  `down_proj` layers (+40% perplexity on their own); attention (+0.1%), MLP gate/up (+1.2%) and the embedding
  table (+0.6%) are nearly harmless. Keeping only `down_proj` in FP32 recovers quality and keeps most of the
  speed-up. These layers receive large activation outliers, which dynamic INT8 squeezes into one scale per tensor.
- **Perplexity is not the whole story.** `int8_mixed` passes the perplexity gate but produces the same greedy
  tokens as PyTorch only 38% of the time, because one early different token changes everything after it.
  Check task-level quality before shipping; the gate here is a minimum bar.
- **The faster engine depends on prompt length.** ONNX FP32 reaches the first token 18% sooner than PyTorch at
  64 prompt tokens but 21% later at 256; decoding is 16-24% faster at both. For long prompts (RAG) check TTFT,
  not just tokens/s.
- **Concurrency buys nothing without batching.** Throughput stays at ~0.57 req/s from 1 to 8 concurrent users
  while p95 grows from 1.9 s to 15 s: the single worker is saturated and extra requests only queue.
- **The SLA is an assumption sized to the workload.** 4 s p95 end-to-end for a non-streamed 256+32-token
  request (a single request alone takes 2.6 s p50 with ONNX FP32 on this CPU). Change it with `--sla-ms`.

## Correctness fixes
Problems found in an audit of the pipeline, each fixed with a regression test:
- **Wrong model measured.** Changing `LAB_MODEL` silently reused the previous model's ONNX export with the new
  tokenizer. Exports now live in one folder per model and every CSV and report records the model.
- **Impossible speeds.** `--new-tokens 1` or timing noise was clamped to 1e-6 ms, reporting ~264,000 tokens/s
  and $0 cost. Decode time is now unclamped and a run fails loudly when it cannot be measured.
- **Two different baselines.** Greedy agreement was measured against the first engine listed, perplexity against
  PyTorch. One shared rule now decides the baseline, and agreement is recomputed against it on every run.
- **Partial re-runs.** Re-running one engine overwrote all other results and compared the engine with itself.
  Earlier results are now kept, but only when model, prompt lengths, output tokens, threads and runs match.
- **Unmeasurable quality passed the gate.** A NaN perplexity compared as "not worse than 5%". It now fails.
- **Benchmark and load test measured different requests** (~128 words / 16 tokens vs 256 / 32 tokens), so cost
  and capacity described different workloads. Both now share one workload and the benchmark's exact prompt;
  capacity per instance went from 0.83 to 0.38 req/s once requests matched.

## Limitations 
- Small model, shared CPU, one instance type: trust ratios more than absolute numbers.
- One worker per instance and no continuous batching; production stacks (vLLM, TGI, Triton) would do better.
- Perplexity on a small text set is a relative quality signal, not a full evaluation.
- Instance and API prices are assumptions: replace them with current numbers.

## Troubleshooting
- `optimum-cli` export fails: run `pip install -U "optimum[onnxruntime]<1.28" "transformers<4.54"` and retry.
- Out of memory during export: use `LAB_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct`.
- INT8 loads but outputs look wrong: report it honestly; the greedy-agreement and perplexity columns will show it.

## Ideas to extend
Static (calibrated) quantization, 4-bit weights, a `batch_size > 1` benchmark, request batching in the server,
comparison against `llama.cpp`, a GPU run on a free Colab T4.
