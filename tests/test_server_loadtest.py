import asyncio
import subprocess
import sys

import httpx
import pandas as pd
from fastapi.testclient import TestClient

from src.config import ROOT

from src.engines import DummyEngine
from src.loadtest import run_level
from src.server import create_app


def test_health_and_generate():
    client = TestClient(create_app(DummyEngine()))
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/health").json()["model"]
    r = client.post("/generate", json={"prompt": "hello world", "max_new_tokens": 5}).json()
    assert r["new_tokens"] == 5 and r["compute_ms"] > 0 and r["queue_ms"] >= 0


def test_validation():
    client = TestClient(create_app(DummyEngine()))
    assert client.post("/generate", json={"prompt": "x", "max_new_tokens": 0}).status_code == 422


def test_queueing_grows_with_concurrency():
    async def go():
        app = create_app(DummyEngine())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as client:
            one = await run_level(client, 1, 1.0, "word " * 20, 8)
            four = await run_level(client, 4, 1.0, "word " * 20, 8)
        return one, four
    one, four = asyncio.run(go())
    assert one["requests"] > 0 and four["requests"] > 0
    assert four["errors"] == 0
    assert four["mean_queue_ms"] > one["mean_queue_ms"]      # single worker => requests queue under load
    assert four["p95_ms"] > one["p95_ms"]


def test_prompt_endpoint_returns_benchmark_prompt():
    engine = DummyEngine()
    r = TestClient(create_app(engine)).get("/prompt", params={"tokens": 12}).json()
    assert r["prompt"] == engine.make_prompt(12) and r["prompt_tokens"] == 12


def test_load_test_records_its_workload(tmp_path):
    out = tmp_path / "load.csv"
    cmd = [sys.executable, "-m", "src.loadtest", "--engine", "dummy", "--levels", "1", "2", "--duration", "1",
           "--prompt-tokens", "20", "--max-new-tokens", "4", "--port", "8799", "--out", str(out)]
    subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True, timeout=120)
    df = pd.read_csv(out)
    assert (df.prompt_tokens == 20).all() and (df.new_tokens == 4).all()
    assert (df.requests > 0).all() and (df.errors == 0).all()
