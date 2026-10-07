# LLM Inference Cost Lab

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
python -m src.loadtest --engine onnx_int8 --levels 1 2 4 8 --duration 40
python -m src.report --hourly-usd 0.085 --sla-ms 3000 --target-rps 5
pytest -q            # works without torch: tests use a simulated engine
```
Faster first try: `export LAB_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct` before step 1.

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
_Fill in after your run (copy the headline from `results/REPORT.md`):_

| Engine | TTFT p50 | Decode tok/s | $/1M output tok | Perplexity | Memory |
|---|---|---|---|---|---|
| pytorch | | | | | |
| onnx_fp32 | | | | | |
| onnx_int8 | | | | | |

**Recommendation:** _one paragraph, e.g. "Deploy INT8 at concurrency N: X req/s per instance within a Y ms p95 SLA, Z instances for 5 req/s, $W/month."_

## Limitations (keep these in your write-up)
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
