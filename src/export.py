import json
import os
import statistics
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from data import get_datasets
from models import SmallCNN

model = SmallCNN()
model.load_state_dict(torch.load("models/cnn_best.pt", map_location="cpu", weights_only=True))
model.eval()                                        # critical: freeze BN stats, disable dropout

_, _, test_ds = get_datasets()
x_test, y_test = test_ds.tensors
sample = x_test[:64]

# ---- TorchScript (tracing) ----
traced = torch.jit.trace(model, sample[:1])
traced.save("models/cnn_ts.pt")
ts_model = torch.jit.load("models/cnn_ts.pt")

# ---- ONNX with a dynamic batch dimension ----
torch.onnx.export(model, sample[:1], "models/cnn.onnx",
                  input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                  opset_version=17)
onnx_model = onnx.load("models/cnn.onnx")
onnx.checker.check_model(onnx_model)
inp = onnx_model.graph.input[0]
dims = [d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]
print(f"ONNX input '{inp.name}' shape {dims}, opset {onnx_model.opset_import[0].version}")

sess = ort.InferenceSession("models/cnn.onnx", providers=["CPUExecutionProvider"])

# ---- numerical parity on 64 images ----
with torch.no_grad():
    ref = model(sample).numpy()
    ts_out = ts_model(sample).numpy()
onnx_out = sess.run(["logits"], {"input": sample.numpy()})[0]
diff_ts = float(np.abs(ref - ts_out).max())
diff_onnx = float(np.abs(ref - onnx_out).max())
print(f"max |torch - torchscript| = {diff_ts:.2e}")
print(f"max |torch - onnx|        = {diff_onnx:.2e}")

# ---- full test-set accuracy through ONNX Runtime ----
correct = 0
xs = x_test.numpy()
for i in range(0, len(xs), 1000):
    out = sess.run(["logits"], {"input": xs[i:i + 1000]})[0]
    correct += int((out.argmax(1) == y_test[i:i + 1000].numpy()).sum())
onnx_acc = correct / len(xs)
print(f"ONNX Runtime test accuracy: {onnx_acc:.4f}")


# ---- single-image latency ----
def bench(fn, n=200):
    for _ in range(20):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


x1 = sample[:1]
x1_np = x1.numpy()
with torch.no_grad():
    torch_ms = bench(lambda: model(x1))
ort_ms = bench(lambda: sess.run(None, {"input": x1_np}))
print(f"median latency (batch 1): torch {torch_ms:.3f} ms | onnxruntime {ort_ms:.3f} ms")

sizes = {p: os.path.getsize(p) for p in ["models/cnn_best.pt", "models/cnn_ts.pt", "models/cnn.onnx"]}
for p, s in sizes.items():
    print(f"{p:22s} {s / 1024:8.1f} KiB")

with open("runs/export_result.json", "w") as f:
    json.dump({"diff_torchscript": diff_ts, "diff_onnx": diff_onnx, "onnx_test_acc": onnx_acc,
               "torch_ms": torch_ms, "ort_ms": ort_ms, "sizes": sizes}, f, indent=2)
