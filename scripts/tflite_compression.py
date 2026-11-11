"""TFLite export of the paper's MLP (128-64-32) with compression variants; measured size, latency, fidelity.

Variants: FP32 | dynamic-range INT8 | full-integer INT8 | 50 % magnitude-pruned + dynamic INT8.
Fidelity = fraction of predictions equal to the FP32 Keras model (on the 7 real recordings and 2000 synthetic
points drawn around them), NOT task accuracy (the model is not statistically above chance, see the paper).
Latency is on this laptop (1 thread); run it under `taskpolicy -b` for efficiency cores. It is not a phone.
"""
import gzip, json, os, statistics, sys, time, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import tensorflow as tf
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from sklearn.preprocessing import StandardScaler
from sprintml.features import FEATURES
from sprintml.labels import assign_tiers, tier_thresholds
from sprintml.train import augment

OUT = ROOT / "artifacts" / "tflite"; OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ROOT / "artifacts" / "features_clean.csv")
y = assign_tiers(df.v_statures_per_s, tier_thresholds(df.v_statures_per_s))
sc = StandardScaler().fit(df[FEATURES].values); X = sc.transform(df[FEATURES].values)
rng = np.random.default_rng(42)
Xa, ya = augment(X, y, rng, 300, 0.05)
tf.keras.utils.set_random_seed(42)

def build():
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
    from tensorflow.keras.regularizers import l2
    m = Sequential([Input((19,))] + [l for u, p in ((128, .4), (64, .3), (32, .2)) for l in
                    (Dense(u, activation="relu", kernel_regularizer=l2(1e-3)), BatchNormalization(), Dropout(p))]
                   + [Dense(3, activation="softmax")])
    m.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"]); return m

model = build(); model.fit(Xa, ya, epochs=60, batch_size=32, verbose=0)

# 50 % magnitude pruning of every Dense kernel + short fine-tune with the mask held
pruned = build(); pruned.set_weights(model.get_weights())
masks = {}
for l in pruned.layers:
    if isinstance(l, tf.keras.layers.Dense):
        w, b = l.get_weights(); thr = np.quantile(np.abs(w), 0.5); mk = (np.abs(w) >= thr).astype("float32")
        masks[l.name] = mk; l.set_weights([w * mk, b])
for _ in range(10):
    pruned.fit(Xa, ya, epochs=1, batch_size=32, verbose=0)
    for l in pruned.layers:
        if l.name in masks:
            w, b = l.get_weights(); l.set_weights([w * masks[l.name], b])
sparsity = np.mean(np.concatenate([(l.get_weights()[0] == 0).ravel() for l in pruned.layers if l.name in masks]))

def convert(m, mode):
    c = tf.lite.TFLiteConverter.from_keras_model(m)
    if mode in ("int8_dynamic", "pruned_int8"):
        c.optimizations = [tf.lite.Optimize.DEFAULT]
    if mode == "int8_full":
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.representative_dataset = lambda: ([Xa[i:i + 1].astype("float32")] for i in rng.choice(len(Xa), 100, replace=False))
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        c.inference_input_type = tf.float32; c.inference_output_type = tf.float32
    return c.convert()

variants = {"fp32": (model, "fp32"), "int8_dynamic": (model, "int8_dynamic"),
            "int8_full": (model, "int8_full"), "pruned_int8": (pruned, "pruned_int8")}
test = np.vstack([X, np.random.default_rng(1).normal(0, 1, (2000, 19)) * 0.5 + X[np.random.default_rng(2).integers(0, len(X), 2000)]]).astype("float32")
ref = model.predict(test, verbose=0).argmax(1)
res = {}
for name, (m, mode) in variants.items():
    blob = convert(m, mode); path = OUT / f"{name}.tflite"; path.write_bytes(blob)
    it = tf.lite.Interpreter(model_path=str(path), num_threads=1); it.allocate_tensors()
    i, o = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
    preds = []
    for r in test:
        it.set_tensor(i, r[None]); it.invoke(); preds.append(it.get_tensor(o).argmax())
    x1 = test[:1]; t = []
    for _ in range(3000):
        t0 = time.perf_counter(); it.set_tensor(i, x1); it.invoke(); it.get_tensor(o); t.append((time.perf_counter() - t0) * 1e6)
    res[name] = {"size_kb": round(len(blob) / 1024, 1), "gzip_kb": round(len(gzip.compress(blob)) / 1024, 1),
                 "latency_us_median": round(statistics.median(t), 1), "latency_us_p95": round(float(np.percentile(t, 95)), 1),
                 "agreement_with_fp32_keras": round(float((np.array(preds) == ref).mean()), 4)}
    print(name, res[name], flush=True)
res["pruned_sparsity"] = round(float(sparsity), 3)
res["keras_params"] = int(model.count_params())
(OUT / "results.json").write_text(json.dumps(res, indent=2))
