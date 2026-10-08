"""Streamlit dashboard: engine comparison, load-test curve, capacity & cost planner, report."""
import numpy as np
import pandas as pd
import streamlit as st

from src.config import BENCH_CSV, DEFAULT_HOURLY_USD, DEFAULT_SLA_MS, LOAD_CSV, REPORT_MD
from src.cost import (break_even_requests_per_day, instances_needed, monthly_api_cost, monthly_self_host_cost)
from src.report import (MAX_PPL_INCREASE_PCT, add_cost_columns, capacity_from_loadtest, load_workload, model_of,
                        pick_baseline, pick_optimized, ppl_change_text, rejected_engines, same_model)

st.set_page_config(page_title="LLM Inference Cost Lab", layout="wide")
st.title("LLM Inference Cost Lab")
st.caption("ONNX Runtime vs PyTorch: latency, tokens, quality and cost, from benchmark to capacity plan.")

if not BENCH_CSV.exists():
    st.error("No benchmark results yet. Run `bash run_all.sh` (or `python -m src.benchmark`).")
    st.stop()

with st.sidebar:
    st.header("Assumptions")
    hourly = st.number_input("Instance price (USD/hour)", value=float(DEFAULT_HOURLY_USD), min_value=0.001, step=0.005,
                             format="%.3f")
    sla_ms = st.number_input("p95 latency SLA (ms)", value=float(DEFAULT_SLA_MS), min_value=100.0, step=100.0)
    headroom = st.slider("Max load per instance", 0.3, 0.95, 0.8)

bench = add_cost_columns(pd.read_csv(BENCH_CSV), hourly)
load = pd.read_csv(LOAD_CSV) if LOAD_CSV.exists() else None
engines = bench.engine.unique().tolist()
st.caption(f"Model: `{model_of(bench)}`")
if load is not None and not same_model(bench, load):
    st.warning(f"The load test ran on `{model_of(load)}`, not on the benchmarked model `{model_of(bench)}`. "
               "It is ignored; re-run `python -m src.loadtest`.")
    load = None

for name, pct in rejected_engines(bench):
    st.warning(f"Quality gate: **{name}**: perplexity {ppl_change_text(pct)} vs {pick_baseline(bench)} "
               f"(limit {MAX_PPL_INCREASE_PCT:g}%). It is excluded from the recommendation even if it is cheaper.")

t1, t2, t3, t4 = st.tabs(["Engines", "Load test", "Capacity & cost planner", "Report"])

with t1:
    lens = sorted(bench.prompt_tokens.unique().tolist())
    plen = st.selectbox("Prompt length (tokens)", lens, index=len(lens) // 2)
    view = bench[bench.prompt_tokens == plen].set_index("engine")
    c1, c2, c3 = st.columns(3)
    c1.caption("Time to first token p50 (ms, lower is better)")
    c1.bar_chart(view["ttft_p50_ms"])
    c2.caption("Decode throughput (tokens/s, higher is better)")
    c2.bar_chart(view["decode_tps"])
    c3.caption("Cost per 1M output tokens (USD, lower is better)")
    c3.bar_chart(view["cost_per_1m_output_tokens_usd"])
    st.subheader("Quality and memory")
    st.dataframe(view[["perplexity", "greedy_match", "rss_mb", "model_mb"]])
    st.subheader("All metrics")
    st.dataframe(bench)

with t2:
    if load is None:
        st.info("No load test yet. Run `python -m src.loadtest --engine <engine>`.")
    else:
        st.caption("Latency rises with concurrency because requests queue for the single inference worker.")
        idx = load.set_index("concurrency")
        a, b = st.columns(2)
        a.caption("Latency p50 / p95 (ms) vs concurrency")
        a.line_chart(idx[["p50_ms", "p95_ms"]])
        b.caption("Throughput (requests/s) vs concurrency")
        b.line_chart(idx[["rps"]])
        st.caption("Where the time goes: queueing vs compute (ms)")
        st.bar_chart(idx[["mean_queue_ms", "mean_compute_ms"]])
        st.dataframe(load)

with t3:
    left, right = st.columns(2)
    with left:
        default_engine = pick_optimized(bench)
        eng = st.selectbox("Engine to deploy", engines, index=engines.index(default_engine))
        row_view = bench[bench.engine == eng]
        lens3 = sorted(row_view.prompt_tokens.unique().tolist())
        measured = load_workload(load[load.engine == eng]) if load is not None else None
        start = lens3.index(measured[0]) if measured and measured[0] in lens3 else len(lens3) // 2
        plen3 = st.selectbox("Typical prompt length", lens3, index=start, key="p3")   # default: load-tested size
        row = row_view[row_view.prompt_tokens == plen3].iloc[0]
        rpd = st.number_input("Requests per day", value=500_000, min_value=1_000, step=50_000)
        peak = st.slider("Peak / average traffic", 1.0, 10.0, 3.0)
    with right:
        in_tok = st.number_input("Avg input tokens", value=float(row.prompt_tokens), min_value=1.0)
        out_tok = st.number_input("Avg output tokens", value=float(row.new_tokens), min_value=1.0)
        in_price = st.number_input("Hosted API input price (USD per 1M tokens) - enter CURRENT price",
                                   value=0.15, min_value=0.0, step=0.01)
        out_price = st.number_input("Hosted API output price (USD per 1M tokens) - enter CURRENT price",
                                    value=0.60, min_value=0.0, step=0.01)

    cap = capacity_from_loadtest(load[load.engine == eng], sla_ms) if load is not None and eng in set(load.engine) else None
    if cap:
        inst_rps, src = cap["rps"], f"measured load test (p95 {cap['p95_ms']:.0f} ms within SLA)"
        workload = load_workload(load[load.engine == eng])
        if workload and workload != (int(in_tok), int(out_tok)):
            st.warning(f"Capacity was measured with {workload[0]} prompt / {workload[1]} generated tokens per request, "
                       f"not the {in_tok:.0f} / {out_tok:.0f} entered here, so the instance count is approximate. "
                       f"Re-run `python -m src.loadtest --prompt-tokens {in_tok:.0f} --max-new-tokens {out_tok:.0f}` "
                       "for an exact figure.")
    else:
        inst_rps, src = 1000.0 / row.e2e_p50_ms, "single-request service time (no load test for this engine)"
        if row.e2e_p95_ms > sla_ms:
            st.warning(f"Single-request p95 ({row.e2e_p95_ms:.0f} ms) already exceeds the {sla_ms:.0f} ms SLA.")
    st.caption(f"Per-instance capacity: **{inst_rps:.2f} req/s**, from {src}.")

    peak_rps = rpd / 86400 * peak
    n = instances_needed(peak_rps, inst_rps, headroom)
    self_cost = monthly_self_host_cost(n, hourly)
    api_cost = monthly_api_cost(rpd, in_tok, out_tok, in_price, out_price)
    free_api = in_tok * in_price + out_tok * out_price <= 0      # a free API never breaks even
    be = None if free_api else break_even_requests_per_day(monthly_self_host_cost(1, hourly), in_tok, out_tok,
                                                           in_price, out_price)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Instances needed", n)
    m2.metric("Self-host / month", f"${self_cost:,.0f}")
    m3.metric("Hosted API / month", f"${api_cost:,.0f}")
    m4.metric("1-instance break-even", "n/a" if be is None else f"{be:,.0f} req/day")
    if self_cost < api_cost:
        st.success("Self-hosting is cheaper at this traffic.")
    else:
        st.info("The hosted API is cheaper at this traffic.")

    grid = np.geomspace(max(rpd / 50, 1000), rpd * 5, 40)
    curve = pd.DataFrame({
        "requests_per_day": grid,
        "self_hosted_usd": [monthly_self_host_cost(instances_needed(r / 86400 * peak, inst_rps, headroom), hourly)
                            for r in grid],
        "hosted_api_usd": [monthly_api_cost(r, in_tok, out_tok, in_price, out_price) for r in grid],
    }).set_index("requests_per_day")
    st.caption("Monthly cost vs traffic (self-host cost rises in steps as instances are added)")
    st.line_chart(curve)
    st.caption("A small open model and a hosted frontier model differ in capability; this is a cost comparison only.")

with t4:
    if REPORT_MD.exists():
        st.markdown(REPORT_MD.read_text(encoding="utf-8"))
    else:
        st.info("Run `python -m src.report` to generate the report.")
