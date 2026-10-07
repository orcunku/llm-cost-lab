"""Cost columns + auto-generated markdown report with a quality-gated recommendation."""
import argparse
import math

import pandas as pd

from .config import BENCH_CSV, DEFAULT_HOURLY_USD, LOAD_CSV, REPORT_MD
from .cost import cost_per_million_tokens, instances_needed, monthly_self_host_cost, request_cost

MAX_PPL_INCREASE_PCT = 5.0   # quality gate: an engine may not raise perplexity by more than this vs the baseline


def add_cost_columns(df, hourly_usd):
    df = df.copy()
    df["cost_per_1m_output_tokens_usd"] = df["decode_tps"].apply(lambda t: cost_per_million_tokens(t, hourly_usd))
    df["cost_per_1m_input_tokens_usd"] = df["prefill_tps"].apply(lambda t: cost_per_million_tokens(t, hourly_usd))
    df["cost_per_1k_requests_usd"] = df["e2e_p50_ms"].apply(lambda ms: request_cost(ms / 1000, hourly_usd) * 1000)
    df["hourly_usd"] = hourly_usd
    return df


def model_of(df):
    """Model the results came from; 'unknown' for CSVs written before the model column existed."""
    if df is None or "model" not in df.columns or df.model.isna().all():
        return "unknown"
    return ", ".join(sorted(df.model.dropna().astype(str).unique()))


def same_model(bench, load):
    """False only when both CSVs name their model and the models differ."""
    a, b = model_of(bench), model_of(load)
    return "unknown" in (a, b) or a == b


def baseline_engine(engines):
    """The reference every engine is compared against (perplexity and greedy output): PyTorch when measured."""
    engines = list(engines)
    return "pytorch" if "pytorch" in engines else engines[0]


def pick_baseline(df):
    return baseline_engine(df.engine.unique())


def _per_engine(df):
    return df.groupby("engine", sort=False).agg(ppl=("perplexity", "first"), tps=("decode_tps", "mean"))


def ppl_increase_pct(df, engine):
    per = _per_engine(df)
    return (per.loc[engine, "ppl"] / per.loc[pick_baseline(df), "ppl"] - 1) * 100


def rejected_engines(df, max_pct=MAX_PPL_INCREASE_PCT):
    """Engines that fail the quality gate, as (engine, perplexity increase in %).
    An increase that could not be measured (NaN) fails too: quality has to be shown, not assumed."""
    base = pick_baseline(df)
    return [(e, ppl_increase_pct(df, e)) for e in _per_engine(df).index
            if e != base and not ppl_increase_pct(df, e) <= max_pct]


def ppl_change_text(pct):
    return f"rises {pct:+.0f}%" if math.isfinite(pct) else "could not be measured"


def pick_optimized(df, max_pct=MAX_PPL_INCREASE_PCT):
    """Fastest engine that passes the quality gate (the baseline itself always passes)."""
    per = _per_engine(df)
    bad = {e for e, _ in rejected_engines(df, max_pct)}
    return per[~per.index.isin(bad)].tps.idxmax()


def capacity_from_loadtest(load_df, sla_ms):
    ok = load_df[(load_df.p95_ms <= sla_ms) & (load_df.errors == 0)]
    if ok.empty:
        return None
    best = ok.sort_values("rps").iloc[-1]
    return {"concurrency": int(best.concurrency), "rps": float(best.rps), "p95_ms": float(best.p95_ms)}


def _md_table(df, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def build_report(bench, load, hourly_usd, sla_ms=3000, target_rps=5.0, headroom=0.8):
    bench = add_cost_columns(bench, hourly_usd)
    base_name, opt_name = pick_baseline(bench), pick_optimized(bench)
    rejected = rejected_engines(bench)
    lens = sorted(bench.prompt_tokens.unique())
    plen = lens[len(lens) // 2]

    def at(engine):
        return bench[(bench.engine == engine) & (bench.prompt_tokens == plen)].iloc[0]

    b, o = at(base_name), at(opt_name)
    out = ["# LLM Inference Cost Report", "",
           f"Model: **`{model_of(bench)}`**.", "",
           f"Assumed instance price: **${hourly_usd:.3f}/hour** (edit with `--hourly-usd`). "
           f"Headline comparison at **{int(plen)} prompt tokens**, {int(b.new_tokens)} generated tokens.", "",
           f"**Quality gate:** an engine is only recommended if its perplexity rises by at most "
           f"{MAX_PPL_INCREASE_PCT:g}% versus `{base_name}`.", ""]
    if not math.isfinite(b.perplexity):
        out += [f"> **Warning:** the perplexity of `{base_name}` could not be measured, so no other engine can pass "
                f"the quality gate. Check the baseline run.", ""]
    out += ["## Headline", ""]

    if opt_name == base_name:
        out.append(f"- No optimized engine passed the quality gate, so `{base_name}` stays the recommendation.")
    else:
        ttft_gain = (1 - o.ttft_p50_ms / b.ttft_p50_ms) * 100
        cost_gain = (1 - o.cost_per_1m_output_tokens_usd / b.cost_per_1m_output_tokens_usd) * 100
        ppl_delta = (o.perplexity / b.perplexity - 1) * 100
        mem_gain = (1 - o.rss_mb / b.rss_mb) * 100 if b.rss_mb > 0 else 0.0
        out += [f"- **{opt_name}** vs **{base_name}**: time-to-first-token **{abs(ttft_gain):.0f}% "
                f"{'faster' if ttft_gain >= 0 else 'slower'}**, decode throughput **{o.decode_tps / b.decode_tps:.2f}x** "
                f"({b.decode_tps:.1f} -> {o.decode_tps:.1f} tokens/s).",
                f"- Cost per 1M output tokens: **${b.cost_per_1m_output_tokens_usd:.2f} -> "
                f"${o.cost_per_1m_output_tokens_usd:.2f}** ({-cost_gain:+.0f}%).",
                f"- Quality: perplexity {b.perplexity:.2f} -> {o.perplexity:.2f} ({ppl_delta:+.1f}%), "
                f"greedy-output agreement with baseline {o.greedy_match * 100:.0f}%.",
                f"- Resident memory: {b.rss_mb:.0f} MB -> {o.rss_mb:.0f} MB ({-mem_gain:+.0f}%)."]
    out.append("")

    if rejected:
        out += ["## Engines rejected by the quality gate", ""]
        for name, pct in rejected:
            r = at(name)
            match = f"{r.greedy_match * 100:.0f}%" if math.isfinite(r.greedy_match) else "an unknown share"
            verdict = ("Cheaper is not better when quality drops this much." if math.isfinite(pct)
                       else "Its quality is unproven, so it is not recommended.")
            out.append(f"- `{name}` is {r.decode_tps / b.decode_tps:.2f}x faster and "
                       f"{(1 - r.cost_per_1m_output_tokens_usd / b.cost_per_1m_output_tokens_usd) * 100:.0f}% cheaper, "
                       f"but perplexity {ppl_change_text(pct)} and only {match} of generated tokens "
                       f"match the baseline. **{verdict}**")
        out.append("")

    out += ["## All engines", "",
            _md_table(bench.round(3), ["engine", "prompt_tokens", "ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms",
                                       "decode_tps", "cost_per_1m_output_tokens_usd", "perplexity", "greedy_match"]), ""]

    if load is not None and not load.empty and not same_model(bench, load):
        out += ["## Capacity", "",
                f"> The load test ran on `{model_of(load)}`, not on the benchmarked model `{model_of(bench)}`, "
                f"so its capacity numbers are left out. Re-run `python -m src.loadtest --engine {opt_name}`.", ""]
    elif load is not None and not load.empty:
        eng = opt_name if opt_name in set(load.engine) else load.engine.iloc[0]
        cap = capacity_from_loadtest(load[load.engine == eng], sla_ms)
        out += ["## Capacity (measured with a real HTTP load test)", "",
                f"Load test engine: `{eng}`.", "",
                _md_table(load.round(2), ["engine", "concurrency", "requests", "rps", "p50_ms", "p95_ms",
                                          "mean_queue_ms", "errors"]), ""]
        if eng != opt_name:
            out += [f"> Note: the recommended engine is `{opt_name}` but the load test ran on `{eng}`. "
                    f"Re-run `python -m src.loadtest --engine {opt_name}` for matching capacity numbers.", ""]
        if cap is None:
            out.append(f"- The p95 latency SLA of {sla_ms:.0f} ms is **not met** by `{eng}` on this hardware, "
                       f"even at concurrency 1. Use a smaller model, fewer output tokens, or faster hardware.")
        else:
            n = instances_needed(target_rps, cap["rps"], headroom)
            out += [f"- Max throughput within a {sla_ms:.0f} ms p95 SLA: **{cap['rps']:.2f} req/s per instance** "
                    f"(concurrency {cap['concurrency']}, p95 {cap['p95_ms']:.0f} ms).",
                    f"- To serve **{target_rps:g} req/s** at {headroom:.0%} max load: **{n} instance(s)**, "
                    f"about **${monthly_self_host_cost(n, hourly_usd):,.0f}/month**."]
        out.append("")

    out += ["## Recommendation", "",
            f"Deploy `{opt_name}`. " + ("It passes the quality gate and is the fastest engine that does. "
                                        if opt_name != base_name else "") +
            "Re-run the benchmark on your production instance type before committing, since absolute numbers are "
            "hardware dependent.", "",
            "## Limits of this study", "",
            "- Single small model, shared CPU, one instance type; use ratios rather than absolute numbers.",
            "- One inference worker per instance and no continuous batching; a production server (vLLM, TGI, "
            "Triton) would raise throughput.",
            "- Perplexity on a small built-in text set is a relative signal, not a full quality evaluation.",
            "- Small models are more sensitive to quantization than large ones; this result may not transfer.",
            "- Instance price is an assumption."]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hourly-usd", type=float, default=DEFAULT_HOURLY_USD)
    ap.add_argument("--sla-ms", type=float, default=3000)
    ap.add_argument("--target-rps", type=float, default=5.0)
    ap.add_argument("--print-choice", action="store_true", help="print the recommended engine name and exit")
    args = ap.parse_args()
    bench = pd.read_csv(BENCH_CSV)
    if args.print_choice:
        print(pick_optimized(bench))
        return
    load = pd.read_csv(LOAD_CSV) if LOAD_CSV.exists() else None
    REPORT_MD.write_text(build_report(bench, load, args.hourly_usd, args.sla_ms, args.target_rps), encoding="utf-8")
    print(f"Saved {REPORT_MD}")


if __name__ == "__main__":
    main()
