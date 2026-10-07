import os
import subprocess
import sys

import pandas as pd
import pytest

from src.config import ROOT


def test_benchmark_pipeline_with_simulated_engines(tmp_path):
    cmd = [sys.executable, "-m", "src.benchmark", "--engines", "dummy", "dummy_fast", "--prompt-lens", "16", "32",
           "--new-tokens", "6", "--runs", "3", "--results-dir", str(tmp_path)]
    env = {**os.environ, "LAB_MODEL": "org/test-model"}
    subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True, env=env)
    df = pd.read_csv(tmp_path / "benchmark.csv")
    assert set(df.engine) == {"dummy", "dummy_fast"}
    assert set(df.model) == {"org/test-model"}           # every row records which model it measured
    assert (df.decode_tps > 0).all() and (df.cost_per_1m_output_tokens_usd > 0).all()
    fast = df[df.engine == "dummy_fast"].decode_tps.mean()
    slow = df[df.engine == "dummy"].decode_tps.mean()
    assert fast > slow                                   # 2x faster simulated engine is measured as faster
    assert df[df.engine == "dummy"].greedy_match.iloc[0] == 1.0
    assert df[df.engine == "dummy_fast"].greedy_match.iloc[0] < 1.0


def test_single_new_token_is_rejected(tmp_path):
    cmd = [sys.executable, "-m", "src.benchmark", "--engines", "dummy", "--prompt-lens", "8", "--new-tokens", "1",
           "--runs", "2", "--results-dir", str(tmp_path)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "at least 2" in r.stderr
    assert not (tmp_path / "benchmark.csv").exists()


def test_decode_speed_is_plausible_with_few_tokens(tmp_path):
    cmd = [sys.executable, "-m", "src.benchmark", "--engines", "dummy", "--prompt-lens", "8", "--new-tokens", "2",
           "--runs", "3", "--results-dir", str(tmp_path)]
    subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True)
    tps = pd.read_csv(tmp_path / "benchmark.csv").decode_tps.iloc[0]
    assert 20 < tps < 250                                # dummy decodes one token per 8 ms (~125 tok/s)


def _bench(tmp_path, *engines, new_tokens=4, runs=2, threads=2):
    cmd = [sys.executable, "-m", "src.benchmark", "--engines", *engines, "--prompt-lens", "8",
           "--new-tokens", str(new_tokens), "--runs", str(runs), "--threads", str(threads),
           "--results-dir", str(tmp_path)]
    subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True)
    return pd.read_csv(tmp_path / "benchmark.csv")


def test_rerunning_one_engine_keeps_the_others_and_compares_to_the_baseline(tmp_path):
    _bench(tmp_path, "dummy", "dummy_fast")
    df = _bench(tmp_path, "dummy_fast")                  # re-run only the optimized engine
    assert list(df.engine) == ["dummy", "dummy_fast"]    # earlier baseline rows kept, baseline first
    assert df[df.engine == "dummy_fast"].greedy_match.iloc[0] < 1.0   # not compared against itself


@pytest.mark.parametrize("changed", [{"new_tokens": 6}, {"threads": 1}, {"runs": 3}])
def test_results_from_other_settings_are_replaced_not_mixed(tmp_path, changed):
    _bench(tmp_path, "dummy", "dummy_fast")
    df = _bench(tmp_path, "dummy_fast", **changed)
    assert list(df.engine) == ["dummy_fast"]


def test_baseline_rule_is_shared_by_benchmark_and_report():
    from src.report import baseline_engine, pick_baseline
    assert baseline_engine(["onnx_int8", "pytorch"]) == "pytorch"     # not simply the first engine listed
    assert baseline_engine(["onnx_fp32", "onnx_int8"]) == "onnx_fp32"
    assert pick_baseline(pd.DataFrame({"engine": ["onnx_int8", "pytorch"]})) == "pytorch"


def test_greedy_match_follows_the_current_baseline(tmp_path):
    import json
    from src.benchmark import MODEL_ID, greedy_path, greedy_probes, greedy_vs_baseline
    for engine, outputs in [("pytorch", [[1, 2, 3, 4]]), ("onnx_fp32", [[1, 2, 3, 4]]), ("onnx_int8", [[1, 2, 9, 9]])]:
        greedy_path(tmp_path, engine).write_text(json.dumps({"model": MODEL_ID, "probes": greedy_probes(),
                                                             "outputs": outputs}))
    assert greedy_vs_baseline(tmp_path, "onnx_int8", "onnx_fp32") == 0.5
    assert greedy_vs_baseline(tmp_path, "onnx_fp32", "pytorch") == 1.0     # baseline switched: recomputed
    greedy_path(tmp_path, "pytorch").write_text(json.dumps({"model": "other/model", "probes": greedy_probes(),
                                                            "outputs": [[1, 2, 3, 4]]}))
    assert pd.isna(greedy_vs_baseline(tmp_path, "onnx_fp32", "pytorch"))  # outputs from another model are ignored


def test_kept_rows_get_greedy_recomputed(tmp_path):
    first = _bench(tmp_path, "dummy", "dummy_fast")
    stale = first.assign(greedy_match=0.123)
    stale.to_csv(tmp_path / "benchmark.csv", index=False)
    df = _bench(tmp_path, "dummy")                       # dummy_fast is kept, not re-run
    kept = df[df.engine == "dummy_fast"].greedy_match.iloc[0]
    assert kept == first[first.engine == "dummy_fast"].greedy_match.iloc[0] != 0.123
