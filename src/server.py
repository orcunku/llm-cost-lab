"""Minimal LLM serving endpoint. One inference worker, so queueing delay under load is real and measurable."""
import argparse
import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import MODEL_ID


class GenRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(16, ge=1, le=256)


def create_app(engine):
    app = FastAPI(title="llm-cost-lab")
    executor = ThreadPoolExecutor(max_workers=1)

    @app.get("/health")
    def health():
        return {"status": "ok", "engine": engine.name, "model": MODEL_ID}

    @app.post("/generate")
    async def generate(req: GenRequest):
        arrived = time.perf_counter()

        def job():
            start = time.perf_counter()
            ids = engine.generate(req.prompt, req.max_new_tokens, req.max_new_tokens)
            end = time.perf_counter()
            return ids, (start - arrived) * 1000, (end - start) * 1000

        ids, queue_ms, compute_ms = await asyncio.get_running_loop().run_in_executor(executor, job)
        return {"new_tokens": len(ids), "queue_ms": queue_ms, "compute_ms": compute_ms}

    return app


def main():
    import uvicorn
    from .engines import build_engine
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="onnx_int8")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()
    engine = build_engine(args.engine, args.threads)
    uvicorn.run(create_app(engine), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
