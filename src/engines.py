"""Inference engines behind one interface: PyTorch, ONNX Runtime FP32/INT8, and a simulated engine for tests."""
import math
import time

from .config import FP32_DIR, INT8_DIR, INT8_PC_DIR, MODEL_ID
from .data import load_eval_texts


class Engine:
    name = "engine"
    model_mb = 0.0

    def count_tokens(self, text): raise NotImplementedError
    def make_prompt(self, n_tokens): raise NotImplementedError
    def generate(self, prompt, max_new_tokens, min_new_tokens=0): raise NotImplementedError
    def perplexity(self, texts): raise NotImplementedError


class DummyEngine(Engine):
    """Simulated engine with deterministic latency; used by tests/CI and to demo the pipeline without a model."""

    def __init__(self, name="dummy", speed=1.0):
        self.name, self.speed, self.model_mb = name, speed, 100.0 / speed

    def count_tokens(self, text): return len(text.split())
    def make_prompt(self, n_tokens): return " ".join(["token"] * n_tokens)

    def generate(self, prompt, max_new_tokens, min_new_tokens=0):
        n_out = max(max_new_tokens, min_new_tokens, 1)
        time.sleep((3 + 0.2 * self.count_tokens(prompt)) / 1000 / self.speed)   # prefill + first token
        time.sleep((n_out - 1) * 8 / 1000 / self.speed)                          # decode
        out = [(i * 7) % 100 for i in range(n_out)]
        if self.speed > 1 and out:
            out[-1] += 1
        return out

    def perplexity(self, texts): return 10.0 if self.speed <= 1 else 10.1


class HFEngine(Engine):
    """Shared logic for models that expose the Hugging Face generate() API."""

    def _init_tokenizer(self):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)

    def count_tokens(self, text): return len(self.tok(text)["input_ids"])

    def make_prompt(self, n_tokens):
        base = " ".join(load_eval_texts() * 20)
        return self.tok.decode(self.tok(base)["input_ids"][:n_tokens])

    def generate(self, prompt, max_new_tokens, min_new_tokens=0):
        enc = self.tok(prompt, return_tensors="pt")
        out = self._generate(enc, max_new_tokens, min_new_tokens)
        return out[0, enc["input_ids"].shape[1]:].tolist()

    def _generate(self, enc, max_new_tokens, min_new_tokens):
        import torch
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        with torch.inference_mode():
            return self.model.generate(**enc, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                                       do_sample=False, pad_token_id=pad)

    def perplexity(self, texts, max_len=256):
        import torch
        nll, count = 0.0, 0
        for text in texts:
            ids = self.tok(text, return_tensors="pt", truncation=True, max_length=max_len)
            with torch.inference_mode():
                logits = self.model(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"]).logits
            target = ids["input_ids"][0, 1:]
            nll += torch.nn.functional.cross_entropy(logits[0, :-1].float(), target, reduction="sum").item()
            count += target.numel()
        return math.exp(nll / count)


class TorchEngine(HFEngine):
    name = "pytorch"

    def __init__(self, threads):
        import torch
        from transformers import AutoModelForCausalLM
        torch.set_num_threads(threads)
        self._init_tokenizer()
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).eval()
        self.model_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1e6


class OrtEngine(HFEngine):
    def __init__(self, name, directory, threads):
        import onnxruntime as ort
        from optimum.onnxruntime import ORTModelForCausalLM
        from .export import primary_files, size_mb
        self.name = name
        self._init_tokenizer()
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.model = ORTModelForCausalLM.from_pretrained(str(directory), use_cache=True, use_io_binding=False,
                                                         provider="CPUExecutionProvider", session_options=so)
        self.model_mb = size_mb(primary_files(directory))


def build_engine(name, threads=2):
    if name == "dummy":
        return DummyEngine("dummy", 1.0)
    if name == "dummy_fast":
        return DummyEngine("dummy_fast", 2.0)
    if name == "pytorch":
        return TorchEngine(threads)
    if name == "onnx_fp32":
        return OrtEngine("onnx_fp32", FP32_DIR, threads)
    if name == "onnx_int8":
        return OrtEngine("onnx_int8", INT8_DIR, threads)
    if name == "onnx_int8_pc":
        return OrtEngine("onnx_int8_pc", INT8_PC_DIR, threads)
    raise ValueError(f"unknown engine: {name}")
