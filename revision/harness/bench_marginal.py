#!/usr/bin/env python3
import time
import numpy as np
from numpy.random import default_rng
REPO = str(__import__('pathlib').Path(__file__).resolve().parents[2])

def extract_defs(path, names):
    src = open(path).read()
    out = {}
    for b in src.split("\ndef ")[1:]:
        name = b.split("(", 1)[0].strip()
        if name in names:
            out[name] = "def " + b
    assert set(names) <= set(out), f"missing {set(names)-set(out)}"
    return "\n".join(out[n] for n in names)
g = {"np": np, "default_rng": default_rng}
exec(extract_defs(f"{REPO}/Marginal/Experiment/marginal_QFTtestfalse.py",
                  ["_sanitize_R", "_nearest_correlation", "sample_bernoulli_copula",
                   "chi_square_bootstrap_test"]), g)
boot = g["chi_square_bootstrap_test"]

rng = default_rng(7)
d = 12
p_true = np.clip(rng.uniform(0.2, 0.8, d), 0.05, 0.95)
for m in (500, 2000):
    Xm = (rng.uniform(size=(m, d)) < p_true).astype(int)
    R = np.corrcoef(Xm, rowvar=False)
    t0 = time.perf_counter()
    for _ in range(3):
        boot(Xm, p_true, R_assumed=R, n_bootstrap=300, seed=5)
    t = (time.perf_counter() - t0) / 3
    print(f"[marginal d=12 m={m:4d}] copula bootstrap B=300: {t*1000:7.1f}ms")
