# LLM Inference Cost Report

Model: **`HuggingFaceTB/SmolLM2-135M-Instruct`**.

Assumed instance price: **$0.085/hour** (edit with `--hourly-usd`). Headline comparison at **256 prompt tokens**, 32 generated tokens.

**Quality gate:** an engine is only recommended if its perplexity rises by at most 5% versus `pytorch`.

## Headline

- **onnx_int8_mixed** vs **pytorch**: time-to-first-token **11% faster**, decode throughput **1.61x** (15.0 -> 24.1 tokens/s).
- Cost per 1M output tokens: **$1.57 -> $0.98** (-38%).
- Quality: perplexity 33.49 -> 34.31 (+2.4%), greedy-output agreement with baseline 38%.
- Resident memory: 857 MB -> 946 MB (+10%).

## Engines rejected by the quality gate

- `onnx_int8` is 2.44x faster and 59% cheaper, but perplexity rises +66% and only 7% of generated tokens match the baseline. **Cheaper is not better when quality drops this much.**
- `onnx_int8_pc` is 1.93x faster and 48% cheaper, but perplexity rises +58% and only 7% of generated tokens match the baseline. **Cheaper is not better when quality drops this much.**

## All engines

| engine | prompt_tokens | ttft_p50_ms | ttft_p95_ms | tpot_p50_ms | decode_tps | cost_per_1m_output_tokens_usd | perplexity | greedy_match |
|---|---|---|---|---|---|---|---|---|
| pytorch | 64 | 197.32 | 279.52 | 66.51 | 15.04 | 1.57 | 33.49 | 1.00 |
| pytorch | 256 | 521.27 | 683.21 | 66.59 | 15.02 | 1.57 | 33.49 | 1.00 |
| onnx_fp32 | 64 | 162.09 | 232.11 | 53.57 | 18.67 | 1.26 | 33.49 | 1.00 |
| onnx_fp32 | 256 | 631.30 | 720.73 | 57.25 | 17.47 | 1.35 | 33.49 | 1.00 |
| onnx_int8 | 64 | 68.47 | 113.82 | 21.58 | 46.34 | 0.51 | 55.66 | 0.07 |
| onnx_int8 | 256 | 299.74 | 394.63 | 27.30 | 36.63 | 0.65 | 55.66 | 0.07 |
| onnx_int8_pc | 64 | 89.26 | 97.91 | 30.50 | 32.79 | 0.72 | 52.90 | 0.07 |
| onnx_int8_pc | 256 | 393.24 | 429.42 | 34.43 | 29.04 | 0.81 | 52.90 | 0.07 |
| onnx_int8_mixed | 64 | 107.58 | 205.26 | 38.35 | 26.08 | 0.91 | 34.31 | 0.38 |
| onnx_int8_mixed | 256 | 461.85 | 547.84 | 41.44 | 24.13 | 0.98 | 34.31 | 0.38 |

## Capacity (measured with a real HTTP load test)

Load test engine: `onnx_int8_mixed`. Each request: 256 prompt tokens, 32 generated tokens.

| engine | concurrency | requests | rps | p50_ms | p95_ms | mean_queue_ms | errors |
|---|---|---|---|---|---|---|---|
| onnx_int8_mixed | 1 | 23 | 0.57 | 1736.45 | 1920.58 | 0.16 | 0 |
| onnx_int8_mixed | 2 | 25 | 0.57 | 3469.00 | 3606.10 | 1667.15 | 0 |
| onnx_int8_mixed | 4 | 27 | 0.58 | 6907.84 | 7029.99 | 4802.57 | 0 |
| onnx_int8_mixed | 8 | 30 | 0.56 | 14188.65 | 15075.62 | 10899.04 | 0 |

- Max throughput within a 4000 ms p95 SLA: **0.57 req/s per instance** (concurrency 2, p95 3606 ms).
- To serve **5 req/s** at 80% max load: **11 instance(s)**, about **$683/month**.

## Recommendation

Deploy `onnx_int8_mixed`. It passes the quality gate and is the fastest engine that does. Re-run the benchmark on your production instance type before committing, since absolute numbers are hardware dependent.

## Limits of this study

- Single small model, shared CPU, one instance type; use ratios rather than absolute numbers.
- One inference worker per instance and no continuous batching; a production server (vLLM, TGI, Triton) would raise throughput.
- Perplexity on a small built-in text set is a relative signal, not a full quality evaluation.
- Small models are more sensitive to quantization than large ones; this result may not transfer.
- Instance price is an assumption.