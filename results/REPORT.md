# LLM Inference Cost Report

Model: **`HuggingFaceTB/SmolLM2-135M-Instruct`**.

Assumed instance price: **$0.085/hour** (edit with `--hourly-usd`). Headline comparison at **256 prompt tokens**, 32 generated tokens.

**Quality gate:** an engine is only recommended if its perplexity rises by at most 5% versus `pytorch`.

## Headline

- **onnx_fp32** vs **pytorch**: time-to-first-token **17% slower**, decode throughput **1.16x** (14.9 -> 17.3 tokens/s).
- Cost per 1M output tokens: **$1.59 -> $1.37** (-14%).
- Quality: perplexity 33.49 -> 33.49 (+0.0%), greedy-output agreement with baseline 100%.
- Resident memory: 858 MB -> 1362 MB (+59%).

## Engines rejected by the quality gate

- `onnx_int8` is 2.43x faster and 59% cheaper, but perplexity rises +66% and only 7% of generated tokens match the baseline. **Cheaper is not better when quality drops this much.**
- `onnx_int8_pc` is 1.82x faster and 45% cheaper, but perplexity rises +58% and only 7% of generated tokens match the baseline. **Cheaper is not better when quality drops this much.**

## All engines

| engine | prompt_tokens | ttft_p50_ms | ttft_p95_ms | tpot_p50_ms | decode_tps | cost_per_1m_output_tokens_usd | perplexity | greedy_match |
|---|---|---|---|---|---|---|---|---|
| pytorch | 64 | 218.89 | 244.06 | 68.84 | 14.53 | 1.62 | 33.49 | 1.00 |
| pytorch | 256 | 564.52 | 632.70 | 67.18 | 14.89 | 1.59 | 33.49 | 1.00 |
| onnx_fp32 | 64 | 166.18 | 213.30 | 56.74 | 17.62 | 1.34 | 33.49 | 1.00 |
| onnx_fp32 | 256 | 661.55 | 764.96 | 57.84 | 17.29 | 1.37 | 33.49 | 1.00 |
| onnx_int8 | 64 | 72.69 | 103.56 | 21.57 | 46.35 | 0.51 | 55.66 | 0.07 |
| onnx_int8 | 256 | 334.23 | 390.01 | 27.68 | 36.13 | 0.65 | 55.66 | 0.07 |
| onnx_int8_pc | 64 | 88.92 | 121.90 | 30.93 | 32.33 | 0.73 | 52.90 | 0.07 |
| onnx_int8_pc | 256 | 395.68 | 540.79 | 36.95 | 27.07 | 0.87 | 52.90 | 0.07 |

## Capacity (measured with a real HTTP load test)

Load test engine: `onnx_fp32`. Each request: 256 prompt tokens, 32 generated tokens.

| engine | concurrency | requests | rps | p50_ms | p95_ms | mean_queue_ms | errors |
|---|---|---|---|---|---|---|---|
| onnx_fp32 | 1 | 15 | 0.38 | 2596.99 | 3100.87 | 0.13 | 0 |
| onnx_fp32 | 2 | 16 | 0.37 | 5369.93 | 5921.52 | 2550.04 | 0 |
| onnx_fp32 | 4 | 19 | 0.38 | 10324.38 | 10964.50 | 7017.28 | 0 |
| onnx_fp32 | 8 | 23 | 0.40 | 19869.59 | 20256.30 | 14587.36 | 0 |

- Max throughput within a 4000 ms p95 SLA: **0.38 req/s per instance** (concurrency 1, p95 3101 ms).
- To serve **5 req/s** at 80% max load: **17 instance(s)**, about **$1,055/month**.

## Recommendation

Deploy `onnx_fp32`. It passes the quality gate and is the fastest engine that does. Re-run the benchmark on your production instance type before committing, since absolute numbers are hardware dependent.

## Limits of this study

- Single small model, shared CPU, one instance type; use ratios rather than absolute numbers.
- One inference worker per instance and no continuous batching; a production server (vLLM, TGI, Triton) would raise throughput.
- Perplexity on a small built-in text set is a relative signal, not a full quality evaluation.
- Small models are more sensitive to quantization than large ones; this result may not transfer.
- Instance price is an assumption.