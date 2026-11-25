import json

from data import CLASSES, load_raw

x, y = load_raw(train=False)
for i in range(5):
    pixels = [round(v * 255) for v in x[i, 0].flatten().tolist()]
    with open(f"runs/sample_{i}.json", "w") as f:
        json.dump({"pixels": pixels}, f)
    print(f"runs/sample_{i}.json  true label: {CLASSES[y[i].item()]}")
