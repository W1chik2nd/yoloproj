#!/usr/bin/env python3
# ---------------------------------------------------------------
# check_onnx_model.py
#
# 用途：
#   1. 打印 ONNX 模型的输入 / 输出信息（含动态轴）
#   2. 随机生成 (batch=1) 测试张量跑一次推理，检查数值
#
# 使用：
#   python check_onnx_model.py model1.onnx model2.onnx ...
#   若不带参数，则默认检查 weights/ 目录下两份 osnet 模型
# ---------------------------------------------------------------

import sys
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path


def _readable_shape(dims):
    out = []
    for d in dims:
        if getattr(d, "dim_value", 0) > 0:
            out.append(str(d.dim_value))
        elif getattr(d, "dim_param", ""):
            out.append(f"动态({d.dim_param})")
        else:
            out.append("动态")
    return out


def _gen_dummy(shape):
    concrete = []
    for idx, dim in enumerate(shape):
        if isinstance(dim, int) and dim > 0:
            concrete.append(dim)
        else:  # 动态轴
            concrete.append(1 if idx == 0 else
                            (_raise := (_ for _ in ()).throw(RuntimeError(
                                "非 batch 维为动态，需手动指定尺寸"))))
    return np.random.randn(*concrete).astype(np.float32)


def inspect(model_path: str):
    print(f"\n===== 检查 {model_path} =====")
    if not Path(model_path).exists():
        print("  文件不存在")
        return

    try:
        model = onnx.load(model_path)
    except Exception as e:
        print(f"  ❌ 载入失败: {e}")
        return

    print("  输入:")
    for inp in model.graph.input:
        shp = _readable_shape(inp.type.tensor_type.shape.dim)
        dt = inp.type.tensor_type.elem_type
        print(f"    {inp.name:15s} 形状={shp}  dtype={dt}")

    print("  输出:")
    for out in model.graph.output:
        shp = _readable_shape(out.type.tensor_type.shape.dim)
        dt = out.type.tensor_type.elem_type
        print(f"    {out.name:15s} 形状={shp}  dtype={dt}")

    try:
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        dummy = _gen_dummy(inp.shape)
        outs = sess.run(None, {inp.name: dummy})
        for i, o in enumerate(outs):
            rng = (o.min(), o.max())
            std = o.std()
            if np.all(o == 0):
                flag = "全零 ❌"
            elif np.isnan(o).any():
                flag = "含 NaN ❌"
            elif std < 1e-6:
                flag = f"方差过小({std:.2e}) ❌"
            else:
                flag = "正常 ✅"
            print(f"  输出{i}: 形状={o.shape}  范围=[{rng[0]:.4f},{rng[1]:.4f}]  {flag}")
    except Exception as e:
        print(f"  ❌ onnxruntime 推理失败: {e}")


if __name__ == "__main__":
    models = sys.argv[1:] if len(sys.argv) > 1 else [
        "weights/osnet_x1_0_msmt17.onnx", 
        "weights/osnet_x0_25_msmt17.onnx"
    ]
    for m in models:
        inspect(m)
