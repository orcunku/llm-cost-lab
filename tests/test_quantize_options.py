import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402
from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: E402

from src.export import lm_head_nodes  # noqa: E402


def _w(name):
    return numpy_helper.from_array(np.random.randn(8, 8).astype(np.float32), name)


def _model(with_subgraph):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8])
    layer = helper.make_node("MatMul", ["x", "w1"], ["h"], name="/layers.0/mlp/MatMul")
    if with_subgraph:   # like a merged decoder: the real work lives inside an If branch
        head = helper.make_node("MatMul", ["h", "w2"], ["out"], name="/lm_head/MatMul")
        branch = helper.make_graph([head], "then", [], [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 8])],
                                   initializer=[])
        other = helper.make_graph([helper.make_node("Identity", ["h"], ["out2"])], "else", [],
                                  [helper.make_tensor_value_info("out2", TensorProto.FLOAT, [1, 8])])
        cond = helper.make_tensor("c", TensorProto.BOOL, [], [True])
        nodes = [layer, helper.make_node("Constant", [], ["cond"], value=cond),
                 helper.make_node("If", ["cond"], ["y"], then_branch=branch, else_branch=other)]
    else:
        nodes = [layer, helper.make_node("MatMul", ["h", "w2"], ["y"], name="/lm_head/MatMul")]
    graph = helper.make_graph(nodes, "g", [x], [y], initializer=[_w("w1"), _w("w2")])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    return m


def _ops(model):
    ops = set()
    def walk(g):
        for n in g.node:
            ops.add(n.op_type)
            for a in n.attribute:
                if a.type == onnx.AttributeProto.GRAPH:
                    walk(a.g)
    walk(model.graph)
    return ops


def test_finds_lm_head_even_inside_subgraphs(tmp_path):
    for sub in (False, True):
        p = tmp_path / f"m{sub}.onnx"
        onnx.save(_model(sub), str(p))
        assert lm_head_nodes(p) == ["/lm_head/MatMul"]


def test_per_channel_quantization_keeps_lm_head_in_fp32(tmp_path):
    src, dst = tmp_path / "fp32.onnx", tmp_path / "int8.onnx"
    onnx.save(_model(False), str(src))
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, per_channel=True,
                     nodes_to_exclude=lm_head_nodes(src))
    ops = _ops(onnx.load(str(dst)))
    assert "MatMulInteger" in ops          # the transformer layer was quantized
    assert "MatMul" in ops                 # the excluded lm_head stayed in float


def _mlp_model():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8])
    nodes = [helper.make_node("MatMul", ["x", "w1"], ["a"], name="/model/layers.0/mlp/up_proj/MatMul"),
             helper.make_node("MatMul", ["a", "w2"], ["b"], name="/model/layers.0/mlp/down_proj/MatMul"),
             helper.make_node("MatMul", ["b", "w3"], ["y"], name="/lm_head/MatMul")]
    graph = helper.make_graph(nodes, "g", [x], [y], initializer=[_w("w1"), _w("w2"), _w("w3")])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    return m


def test_mixed_variant_keeps_down_proj_and_lm_head_in_fp32(tmp_path, monkeypatch):
    import src.export as export
    fp32, out = tmp_path / "fp32", tmp_path / "int8_mixed"
    fp32.mkdir()
    onnx.save(_mlp_model(), str(fp32 / "model.onnx"))
    (fp32 / "config.json").write_text("{}")
    monkeypatch.setattr(export, "FP32_DIR", fp32)
    monkeypatch.setitem(export.VARIANTS, "int8_mixed", {**export.VARIANTS["int8_mixed"], "directory": out})
    export.quantize_variant("int8_mixed")
    nodes = {n.name: n.op_type for n in onnx.load(str(out / "model.onnx")).graph.node}
    assert nodes["/model/layers.0/mlp/down_proj/MatMul"] == "MatMul"     # left in float
    assert nodes["/lm_head/MatMul"] == "MatMul"
    assert "/model/layers.0/mlp/up_proj/MatMul" not in nodes             # replaced by an integer MatMul
    assert "MatMulInteger" in nodes.values()
    assert (out / "config.json").exists()                                # loader files copied
