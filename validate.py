import json
import os
import urllib.request

os.chdir(os.path.expanduser("~/dl-lab"))
results = []


def check(name, fn):
    try:
        ok, detail = fn()
    except Exception as e:                       # missing file, server down, etc.
        ok, detail = False, f"{type(e).__name__}: {e}"
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} -- {detail}")


def load(p):
    with open(p) as f:
        return json.load(f)


def t1():
    r = load("runs/task1_autograd.json")
    ok = (r["max_grad_diff"] < 1e-8 and abs(r["b"] - 1.0) < 0.05
          and all(abs(a - b) < 0.05 for a, b in zip(r["w"], [2.0, -3.0, 0.5])))
    return ok, f"grad diff {r['max_grad_diff']:.1e}, w {[round(v, 3) for v in r['w']]}"


def t2():
    r = load("runs/task2_data.json")
    ok = (r["sizes"] == {"train": 55000, "val": 5000, "test": 10000}
          and abs(r["mean"] - 0.2860) < 0.002 and abs(r["std"] - 0.3530) < 0.002)
    return ok, f"sizes {r['sizes']}, mean {r['mean']:.4f}, std {r['std']:.4f}"


def t3():
    r = load("runs/mlp_result.json")
    ok = r["params"] == 235146 and r["test_acc"] >= 0.85 and os.path.exists("models/mlp_best.pt")
    return ok, f"MLP test acc {r['test_acc']:.4f}"


def t4():
    g, f = load("runs/grad_flow.json"), load("runs/task4_fixed.json")
    ok = (g["sigmoid"]["ratio"] < 1e-3 * g["relu_kaiming"]["ratio"]
          and f["val_acc"] >= 0.82 and f["eval_deterministic"] is True)
    return ok, (f"sigmoid ratio {g['sigmoid']['ratio']:.1e}, fixed val acc {f['val_acc']:.4f}, "
                f"deterministic {f['eval_deterministic']}")


def t5():
    r = load("runs/regularization.json")
    ok = (r["dropout_wd"]["gap"] < r["no_reg"]["gap"]
          and r["no_reg"]["final_val_loss"] > 1.2 * r["no_reg"]["best_val_loss"]
          and r["early_stopping"]["stopped_epoch"] < 40)
    return ok, (f"gap {r['no_reg']['gap']:.3f} -> {r['dropout_wd']['gap']:.3f}, "
                f"early stop at {r['early_stopping']['stopped_epoch']}")


def t6():
    c, m = load("runs/cnn_result.json"), load("runs/mlp_result.json")
    ok = (c["params"] == 218682 and c["test_acc"] >= 0.89 and c["test_acc"] > m["test_acc"]
          and os.path.getsize("runs/curves.png") > 0)
    return ok, f"CNN test acc {c['test_acc']:.4f} (MLP {m['test_acc']:.4f})"


def t7():
    r = load("runs/lstm_result.json")
    ok = r["test_mse"] < 0.6 * r["naive_mse"] and r["test_mse"] < 0.02
    return ok, f"LSTM MSE {r['test_mse']:.4f} vs naive {r['naive_mse']:.4f}"


def t8_export():
    e, c = load("runs/export_result.json"), load("runs/cnn_result.json")
    ok = (e["diff_onnx"] < 1e-4 and e["diff_torchscript"] < 1e-4
          and abs(e["onnx_test_acc"] - c["test_acc"]) < 0.002)
    return ok, f"onnx diff {e['diff_onnx']:.1e}, onnx acc {e['onnx_test_acc']:.4f}"


def t8_api():
    with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as resp:
        h = json.load(resp)
    with open("runs/sample_0.json", "rb") as f:
        body = f.read()
    req = urllib.request.Request("http://127.0.0.1:8000/predict", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        p = json.load(resp)
    return h.get("status") == "ok" and p["class_name"] == "Ankle boot", \
        f"health {h.get('status')}, sample_0 -> {p['class_name']} ({p['confidence']})"


check("Task 1 autograd and manual gradient descent", t1)
check("Task 2 data pipeline and statistics", t2)
check("Task 3 training engine and MLP baseline", t3)
check("Task 4 gradient flow and repaired script", t4)
check("Task 5 regularisation and early stopping", t5)
check("Task 6 CNN architecture and accuracy", t6)
check("Task 7 LSTM beats naive baseline", t7)
check("Task 8 TorchScript/ONNX export parity", t8_export)
check("Task 8 REST inference API", t8_api)
print(f"\n{sum(results)}/{len(results)} checks passed")
