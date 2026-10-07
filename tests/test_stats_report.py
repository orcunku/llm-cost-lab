import pandas as pd
from src.report import build_report, capacity_from_loadtest
from src.stats import summarize


def bench_df():
    def row(engine, ttft, tps, ppl, match, rss):
        return dict(engine=engine, prompt_tokens=64, new_tokens=32, runs=5, ttft_p50_ms=ttft, ttft_p95_ms=ttft * 1.2,
                    tpot_p50_ms=1000 / tps, tpot_p95_ms=1100 / tps, e2e_p50_ms=ttft + 31000 / tps,
                    e2e_p95_ms=ttft * 1.2 + 33000 / tps, decode_tps=tps, prefill_tps=64000 / ttft,
                    rss_mb=rss, model_mb=rss / 2, threads=2, perplexity=ppl, greedy_match=match)
    return pd.DataFrame([row("pytorch", 400, 10, 12.0, 1.0, 3000), row("onnx_int8", 200, 25, 12.3, 0.9, 1200)])


def load_df():
    return pd.DataFrame([
        dict(engine="onnx_int8", concurrency=1, requests=20, errors=0, rps=1.5, p50_ms=600, p95_ms=700,
             p99_ms=800, mean_queue_ms=1, mean_compute_ms=600),
        dict(engine="onnx_int8", concurrency=4, requests=22, errors=0, rps=1.6, p50_ms=2400, p95_ms=2900,
             p99_ms=3000, mean_queue_ms=1800, mean_compute_ms=600),
        dict(engine="onnx_int8", concurrency=8, requests=22, errors=0, rps=1.6, p50_ms=4800, p95_ms=5200,
             p99_ms=5300, mean_queue_ms=4200, mean_compute_ms=600)])


def test_summarize():
    s = summarize([1, 2, 3, 4, 5])
    assert s["p50"] == 3 and s["mean"] == 3


def test_capacity_respects_sla():
    cap = capacity_from_loadtest(load_df(), 3000)
    assert cap["concurrency"] == 4
    assert capacity_from_loadtest(load_df(), 100) is None


def test_report_contents():
    text = build_report(bench_df(), load_df(), hourly_usd=0.085, sla_ms=3000, target_rps=5)
    assert "onnx_int8" in text and "pytorch" in text
    assert "2.50x" in text                      # 10 -> 25 tokens/s
    assert "instance(s)" in text and "Recommendation" in text
    assert "faster" in text


def test_report_without_loadtest():
    assert "Capacity" not in build_report(bench_df(), None, hourly_usd=0.085)
