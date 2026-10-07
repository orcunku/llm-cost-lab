#!/usr/bin/env bash
# Full pipeline. Each step can also be run on its own (see README).
set -e
python -m src.export                       # 1. export to ONNX + INT8 quantization
python -m src.benchmark                    # 2. TTFT / tokens-per-sec / quality / memory per engine
python -m src.loadtest --engine onnx_int8  # 3. real HTTP load test (latency vs concurrency)
python -m src.report                       # 4. results/REPORT.md with the headline numbers
streamlit run app.py --server.port 8501 --server.address 0.0.0.0   # 5. dashboard
