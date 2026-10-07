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


def real_world_df():
    """Numbers from an actual Codespace run: INT8 was fast but badly degraded."""
    def row(engine, plen, ttft, tps, ppl, match):
        return dict(engine=engine, prompt_tokens=plen, new_tokens=16, runs=3, ttft_p50_ms=ttft, ttft_p95_ms=ttft * 1.1,
                    tpot_p50_ms=1000 / tps, tpot_p95_ms=1100 / tps, e2e_p50_ms=ttft + 15000 / tps,
                    e2e_p95_ms=ttft * 1.1 + 16000 / tps, decode_tps=tps, prefill_tps=plen * 1000 / ttft,
                    rss_mb=800, model_mb=500, threads=2, perplexity=ppl, greedy_match=match)
    return pd.DataFrame([row("pytorch", 32, 199, 16.07, 33.49, 1.0), row("pytorch", 64, 190, 15.96, 33.49, 1.0),
                         row("onnx_fp32", 32, 102, 17.35, 33.49, 1.0), row("onnx_fp32", 64, 169, 16.86, 33.49, 1.0),
                         row("onnx_int8", 32, 43.5, 43.17, 55.67, 0.067), row("onnx_int8", 64, 71.8, 40.19, 55.67, 0.067)])


def test_quality_gate_rejects_degraded_int8():
    from src.report import pick_optimized, rejected_engines
    df = real_world_df()
    assert [e for e, _ in rejected_engines(df)] == ["onnx_int8"]
    assert pick_optimized(df) == "onnx_fp32"              # fastest engine that keeps quality


def test_report_explains_rejection():
    text = build_report(real_world_df(), None, hourly_usd=0.085)
    assert "rejected by the quality gate" in text
    assert "onnx_int8" in text and "Cheaper is not better" in text
    assert "Deploy `onnx_fp32`" in text


def test_better_int8_variant_is_chosen_when_it_passes():
    from src.report import pick_optimized
    df = real_world_df()
    good = df[df.engine == "onnx_int8"].copy()
    good["engine"], good["perplexity"], good["greedy_match"] = "onnx_int8_pc", 34.2, 0.8   # +2.1%
    assert pick_optimized(pd.concat([df, good], ignore_index=True)) == "onnx_int8_pc"


def test_baseline_kept_when_nothing_passes():
    from src.report import pick_optimized
    df = real_world_df()
    df = df[df.engine != "onnx_fp32"]
    assert pick_optimized(df) == "pytorch"
    assert "No optimized engine passed" in build_report(df, None, hourly_usd=0.085)
