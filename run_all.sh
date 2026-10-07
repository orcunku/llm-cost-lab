#!/usr/bin/env bash
# Full pipeline. Each step can also be run on its own (see README).
set -e
python -m src.export                       # 1. export to ONNX + two INT8 variants
python -m src.benchmark                    # 2. TTFT / tokens-per-sec / quality / memory per engine
ENGINE=$(python -m src.report --print-choice)   # best engine that passes the quality gate
echo "Load testing the recommended engine: $ENGINE"
python -m src.loadtest --engine "$ENGINE"  # 3. real HTTP load test (latency vs concurrency)
python -m src.report                       # 4. results/REPORT.md
streamlit run app.py --server.port 8501 --server.address 0.0.0.0   # 5. dashboard
