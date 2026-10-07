"""Export the LLM to ONNX (with KV-cache) and create a dynamically quantized INT8 copy."""
import shutil
import subprocess
import sys
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

from .config import FP32_DIR, INT8_DIR, MODEL_ID


def primary_files(directory: Path):
    """The ONNX file(s) the optimum loader will actually use."""
    for name in ("model.onnx", "decoder_model_merged.onnx"):
        if (directory / name).exists():
            return [directory / name]
    return sorted(directory.glob("decoder*.onnx"))


def size_mb(files):
    total = 0
    for f in files:
        total += f.stat().st_size
        for extra in f.parent.glob(f.name + "_data*"):
            total += extra.stat().st_size
        for extra in f.parent.glob(f.stem + ".onnx_data"):
            total += extra.stat().st_size
    return total / 1e6


def export_fp32():
    if primary_files(FP32_DIR):
        print("FP32 ONNX export already exists, skipping.")
        return
    FP32_DIR.mkdir(parents=True, exist_ok=True)
    cmd = ["optimum-cli", "export", "onnx", "--model", MODEL_ID, "--task", "text-generation-with-past",
           str(FP32_DIR)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def quantize_int8():
    if primary_files(INT8_DIR):
        print("INT8 model already exists, skipping.")
        return
    INT8_DIR.mkdir(parents=True, exist_ok=True)
    for src in primary_files(FP32_DIR):
        big = size_mb([src]) > 1500
        print(f"Quantizing {src.name} ({size_mb([src]):.0f} MB) ...")
        quantize_dynamic(str(src), str(INT8_DIR / src.name), weight_type=QuantType.QInt8,
                         use_external_data_format=big)
    for f in FP32_DIR.iterdir():  # copy configs / tokenizer files so the loader works
        if f.suffix not in (".onnx",) and ".onnx_data" not in f.name and f.is_file():
            shutil.copy(f, INT8_DIR / f.name)
    print(f"INT8 size: {size_mb(primary_files(INT8_DIR)):.0f} MB vs FP32 {size_mb(primary_files(FP32_DIR)):.0f} MB")


if __name__ == "__main__":
    export_fp32()
    quantize_int8()
