import subprocess
import sys

import pandas as pd

from src.config import ROOT


def test_benchmark_pipeline_with_simulated_engines(tmp_path):
    cmd = [sys.executable, "-m", "src.benchmark", "--engines", "dummy", "dummy_fast", "--prompt-lens", "16", "32",
           "--new-tokens", "6", "--runs", "3", "--results-dir", str(tmp_path)]
    subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True)
    df = pd.read_csv(tmp_path / "benchmark.csv")
    assert set(df.engine) == {"dummy", "dummy_fast"}
    assert (df.decode_tps > 0).all() and (df.cost_per_1m_output_tokens_usd > 0).all()
    fast = df[df.engine == "dummy_fast"].decode_tps.mean()
    slow = df[df.engine == "dummy"].decode_tps.mean()
    assert fast > slow                                   # 2x faster simulated engine is measured as faster
    assert df[df.engine == "dummy"].greedy_match.iloc[0] == 1.0
    assert df[df.engine == "dummy_fast"].greedy_match.iloc[0] < 1.0
