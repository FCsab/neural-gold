"""Is the network the bottleneck? LightGBM on the 5-minute dataset's rows.

Same rows, the same 70/15/15 chronological split and the same purge as
``training.train_model``, the current row's features only (no window). It also
splits the score into its two parts: telling a barrier hit from a timeout
(volatility) and telling up from stop among the hits (direction). A long-only
trade is paid only for direction.

    .venv/bin/python scripts/direction_check.py   # needs data/dataset.parquet
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score

from crypto_neural.training import purge_overlap, balanced_accuracy

d = pd.read_parquet("data/dataset.parquet")
d = d[d["valid"]].reset_index(drop=True)
feats = [c for c in d.columns if c not in {"open_time","close","label","barrier_up","barrier_down",
         "exit_index","exit_time","exit_price","realized_return","holding_bars","valid"}]
n = len(d); a, b = int(n*.70), int(n*.85)
tr, va, te = np.arange(a), np.arange(a, b), np.arange(b, n)
tr, va, _ = purge_overlap(d, tr, va, te)
X, y = d[feats].to_numpy(np.float32), d["label"].to_numpy() + 1  # 0=stop 1=timeout 2=up
w = len(y[tr]) / (3 * np.bincount(y[tr]))
m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=63, min_child_samples=500,
                       subsample=0.7, subsample_freq=1, colsample_bytree=0.7, verbose=-1)
m.fit(X[tr], y[tr], sample_weight=w[y[tr]], eval_X=X[va], eval_y=y[va],
      eval_sample_weight=[w[y[va]]], callbacks=[lgb.early_stopping(100, verbose=False)])
p = m.predict_proba(X[te]); pred = p.argmax(1)
bal, chance = balanced_accuracy(y[te] - 1, pred - 1)
print(f"test rows {len(te):,}  trees {m.best_iteration_}")
print(f"LightGBM balanced accuracy {bal:.4f} (chance {chance:.4f})")
hit = y[te] != 1
print(f"AUC barrier-hit vs timeout (volatility skill): {roc_auc_score(hit, 1 - p[:,1]):.4f}")
both = hit
up = (y[te][both] == 2)
score = p[both, 2] / (p[both, 2] + p[both, 0])
print(f"AUC up vs stop among barrier hits (DIRECTION skill): {roc_auc_score(up, score):.4f}")
imp = sorted(zip(m.booster_.feature_importance('gain'), feats), reverse=True)[:6]
print("top features:", [f for _, f in imp])
