"""Export the LLM to ONNX (with KV-cache) and create three INT8 variants:
  int8       : naive dynamic quantization (per-tensor weights)
  int8_pc    : per-channel weights, output layer (lm_head) kept in FP32
  int8_mixed : like int8_pc, but the MLP down_proj layers and the embedding table also stay in FP32.
               down_proj inputs carry large activation outliers that per-tensor dynamic activation
               quantization cannot represent; on SmolLM2-135M they cause almost all of the quality loss."""
import shutil
import subprocess
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

from .config import FP32_DIR, INT8_DIR, INT8_MIXED_DIR, INT8_PC_DIR, MODEL_ID


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
        for extra in set(f.parent.glob(f.name + "_data*")) | set(f.parent.glob(f.stem + ".onnx_data*")):
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


def _collect_matmuls(graph, needle, found):
    for node in graph.node:
        if node.op_type == "MatMul" and needle in node.name:
            found.append(node.name)
        for attr in node.attribute:                       # merged decoders keep layers inside If-subgraphs
            if attr.type == onnx.AttributeProto.GRAPH:
                _collect_matmuls(attr.g, needle, found)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for g in attr.graphs:
                    _collect_matmuls(g, needle, found)


def matmul_nodes(model_path: Path, needle: str):
    """Names of the MatMul nodes whose name contains `needle` (e.g. "lm_head", "down_proj")."""
    model = onnx.load(str(model_path), load_external_data=False)
    found = []
    _collect_matmuls(model.graph, needle, found)
    return sorted(set(found))


def lm_head_nodes(model_path: Path):
    return matmul_nodes(model_path, "lm_head")


# keep_fp32: MatMul name patterns left unquantized. op_types: None quantizes ORT's defaults (MatMul and the
# embedding Gather); ["MatMul"] leaves the embedding table in FP32.
VARIANTS = {
    "int8": dict(directory=INT8_DIR, per_channel=False, keep_fp32=(), op_types=None),
    "int8_pc": dict(directory=INT8_PC_DIR, per_channel=True, keep_fp32=("lm_head",), op_types=None),
    "int8_mixed": dict(directory=INT8_MIXED_DIR, per_channel=True, keep_fp32=("lm_head", "down_proj"),
                       op_types=["MatMul"]),
}


def quantize_variant(variant: str):
    cfg = VARIANTS[variant]
    out_dir: Path = cfg["directory"]
    if primary_files(out_dir):
        print(f"{variant}: already exists, skipping.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in primary_files(FP32_DIR):
        exclude = []
        for needle in cfg["keep_fp32"]:
            found = matmul_nodes(src, needle)
            print(f"{variant}: keeping {len(found)} {needle} node(s) in FP32" + ("" if found else " (none found!)"))
            exclude += found
        print(f"{variant}: quantizing {src.name} ({size_mb([src]):.0f} MB) ...")
        quantize_dynamic(str(src), str(out_dir / src.name), weight_type=QuantType.QInt8,
                         per_channel=cfg["per_channel"], nodes_to_exclude=exclude,
                         op_types_to_quantize=cfg["op_types"], use_external_data_format=size_mb([src]) > 1500)
    for f in FP32_DIR.iterdir():  # copy configs / tokenizer files so the loader works
        if f.is_file() and f.suffix != ".onnx" and ".onnx_data" not in f.name:
            shutil.copy(f, out_dir / f.name)
    print(f"{variant}: {size_mb(primary_files(out_dir)):.0f} MB vs FP32 {size_mb(primary_files(FP32_DIR)):.0f} MB")


if __name__ == "__main__":
    export_fp32()
    for v in VARIANTS:
        quantize_variant(v)
