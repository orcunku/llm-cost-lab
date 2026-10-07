# LLM Inference Cost Report

Assumed instance price: **$0.085/hour** (edit with `--hourly-usd`). Headline comparison at **64 prompt tokens**, 16 generated tokens.

**Quality gate:** an engine is only recommended if its perplexity rises by at most 5% versus `pytorch`.

## Headline

- **onnx_fp32** vs **pytorch**: time-to-first-token **12% faster**, decode throughput **1.17x** (16.1 -> 18.8 tokens/s).
- Cost per 1M output tokens: **$1.47 -> $1.25** (-14%).
- Quality: perplexity 33.49 -> 33.49 (+0.0%), greedy-output agreement with baseline 100%.
- Resident memory: 855 MB -> 1351 MB (+58%).

## Engines rejected by the quality gate

- `onnx_int8` is 2.92x faster and 66% cheaper, but perplexity rises +66% and only 7% of generated tokens match the baseline. **Cheaper is not better when quality drops this much.**
- `onnx_int8_pc` is 2.01x faster and 50% cheaper, but perplexity rises +58% and only 7% of generated tokens match the baseline. **Cheaper is not better when quality drops this much.**

## All engines

| engine | prompt_tokens | ttft_p50_ms | ttft_p95_ms | tpot_p50_ms | decode_tps | cost_per_1m_output_tokens_usd | perplexity | greedy_match |
|---|---|---|---|---|---|---|---|---|
| pytorch | 32 | 209.91 | 245.37 | 57.11 | 17.51 | 1.35 | 33.49 | 1.00 |
| pytorch | 64 | 185.34 | 237.80 | 62.12 | 16.10 | 1.47 | 33.49 | 1.00 |
| onnx_fp32 | 32 | 94.60 | 115.59 | 55.76 | 17.93 | 1.32 | 33.49 | 1.00 |
| onnx_fp32 | 64 | 163.35 | 252.67 | 53.12 | 18.82 | 1.25 | 33.49 | 1.00 |
| onnx_int8 | 32 | 44.00 | 64.22 | 22.45 | 44.54 | 0.53 | 55.66 | 0.07 |
| onnx_int8 | 64 | 68.93 | 128.23 | 21.28 | 46.98 | 0.50 | 55.66 | 0.07 |
| onnx_int8_pc | 32 | 62.10 | 95.60 | 29.89 | 33.45 | 0.71 | 52.90 | 0.07 |
| onnx_int8_pc | 64 | 89.79 | 110.03 | 30.96 | 32.30 | 0.73 | 52.90 | 0.07 |

## Capacity (measured with a real HTTP load test)

Load test engine: `onnx_fp32`.

| engine | concurrency | requests | rps | p50_ms | p95_ms | mean_queue_ms | errors |
|---|---|---|---|---|---|---|---|
| onnx_fp32 | 1 | 17 | 0.82 | 1223.30 | 1326.44 | 0.23 | 0 |
| onnx_fp32 | 2 | 18 | 0.83 | 2380.67 | 2569.29 | 1118.92 | 0 |
| onnx_fp32 | 4 | 20 | 0.84 | 4751.47 | 4839.74 | 3215.85 | 0 |
| onnx_fp32 | 8 | 24 | 0.84 | 9510.31 | 9625.68 | 6955.69 | 0 |

- Max throughput within a 3000 ms p95 SLA: **0.83 req/s per instance** (concurrency 2, p95 2569 ms).
- To serve **5 req/s** at 80% max load: **8 instance(s)**, about **$496/month**.

## Recommendation

Deploy `onnx_fp32`. It passes the quality gate and is the fastest engine that does. Re-run the benchmark on your production instance type before committing, since absolute numbers are hardware dependent.

## Limits of this study

- Single small model, shared CPU, one instance type; use ratios rather than absolute numbers.
- One inference worker per instance and no continuous batching; a production server (vLLM, TGI, Triton) would raise throughput.
- Perplexity on a small built-in text set is a relative signal, not a full quality evaluation.
- Small models are more sensitive to quantization than large ones; this result may not transfer.
- Instance price is an assumption.