# Training

Build-order step 2, partial: the model, the training loop, the registry, and
the UI around them. What is *not* here: the compare view, shadow/live
promotion, and the predictions log — those belong to the execution phase.

## How it runs

Training is started from the **Training** tab, or from the API:

```bash
curl -k -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"minutes": 10, "epochs": 200, "window": 64}' \
  https://127.0.0.1:8443/api/train/start
```

Two ways to bound a run, and you can use either or both:

- **minutes** — wall-clock budget. When it expires the loop stops at the next
  batch boundary, evaluates the test slice, and writes a checkpoint.
- **max epochs** — the classic bound.

Leave **minutes blank** (or `null` in the API) and the run continues until you
press **Stop**. Stop is cooperative: the child checks the flag between
batches, so it exits within one batch, not one epoch. Whatever it learned is
saved.

Either way you get a model. A time-boxed or stopped run is a *stopped* run,
not a failed one: the checkpoint, metrics, and registry row are all written.

## Where it runs

The loop runs in a spawned child process, not a thread and not the event loop.

- PyTorch holds the GIL across long stretches of Python-level work; a thread
  would make the UI stutter.
- A process is a hard boundary. If the GPU driver page-faults, the child dies
  and the job reports an error instead of taking the server with it.
- The parent talks to the child through a `multiprocessing.Queue` (progress)
  and an `Event` (stop). The child never touches SQLite — the parent writes
  the registry. That keeps the plan's single-writer rule without a queue
  table.

## Device selection

The plan's rule is followed literally: `torch.cuda.is_available()` is not
trusted. The child runs a real forward/backward/synchronize probe on the GPU
and falls back to CPU with a recorded reason if anything raises.

On this machine (RX 7600 XT, gfx1100, ROCm 6.4 wheels) the probe passes and
training runs on the GPU. If the driver starts misbehaving, set `device` to
`cpu` in the UI — the escape hatch exists so a bad GPU night does not block
work.

The API process never imports torch. Availability is checked with
`importlib.util.find_spec`, so the server starts without a 300 MB import
unless you actually train.

## Data and splits

Training reads `data/dataset.parquet`, built from the Dataset tab. Feature
names come from `data/reports/dataset.json`, falling back to the parquet
schema.

- Windows of `window` rows per sample, label taken at the last row.
- Chronological 70 / 15 / 15 split — no shuffling across time.
- Standardisation statistics come from the **training slice only** and are
  stored in the checkpoint.
- Class weights are computed from the training slice, so the majority
  "stop" class does not dominate the loss.

Windows are gathered with `sliding_window_view`, so a 210k-row dataset is a
view, not a 1 GB copy.

## What a run reports

Per-class precision / recall / F1, a 3×3 confusion matrix, balanced accuracy,
its constant-guess baseline, plain accuracy, expected realised return per
predicted class, the signal edge (mean return of longs minus mean return of
stops), long coverage, and the validation-loss curve.

Accuracy alone is never the promotion criterion: a model that always predicts
"timeout" can look respectable and have no edge.

The comparison has to be like with like. Balanced accuracy is the mean recall
across the three outcomes, so a constant predictor that always answers one
class scores **1/3** on it — not the 48% plain accuracy that same predictor
scores when one class holds 48% of the rows. Comparing balanced accuracy
against the plain majority share (an earlier version of this code did exactly
that) understates every model by 15 points and turns "at chance" into
"catastrophically wrong".

A run is marked **candidate** only when it beats the constant guess on
balanced accuracy by more than 5 points; anything less lands in the registry
as **draft**, with its metrics attached so the result is inspectable rather
than discarded.

## Measured on this machine

Two-minute budget, 210,322 rows, window 64, batch 512, 21 features, GPU:

```
device               cuda (AMD Radeon Graphics, probe passed)
epochs               10 in 121.2 s   (batch 512)
balanced accuracy    0.403
constant-guess       0.333
plain accuracy       0.417
always-guess accuracy 0.485
signal edge          +0.006%
test rows            31,549
parameters           105,795
status               draft
```

A second run with `max_holding_bars=24`, batch 256, patience 8:

```
epochs               11 in 328 s, best epoch 3
balanced accuracy    0.3575
constant-guess       0.3333   (+2.4 points -> marginal)
plain accuracy       0.4038
always-guess         0.4838
recall   stop 41.2%  timeout 14.6%  target 51.5%
status               draft
```

That run was then repeated with the two changes the curve seemed to call for —
a smaller learning rate (1e-4) and twice the context (window 128), patience 5:

```
epochs               11 in 757 s, best epoch 6
balanced accuracy    0.3528
constant-guess       0.3333   (+1.9 points)
plain accuracy       0.3902
validation floor     1.0616   (previous run: 1.0619)
status               draft
```

**Neither change helped.** The validation floor moved by 0.0003 while the
wall clock more than doubled, and balanced accuracy landed half a point lower
— noise in either direction. When two very different step sizes and window
lengths converge to the same loss, those are not the binding constraint; the
feature set and the amount of history are. The defaults stay at window 64 /
lr 3e-4, which reaches the same place in half the time, with patience 10 to
give a slower start room to recover.

What the model does find: 51.5% of real targets against a 32% base rate, while
losing the timeout class almost entirely (14.6% recall). Training loss falls
monotonically as validation flattens and turns up — a small model beginning to
memorise, not an undertrained one.

Stopping an unbounded run: start → 1 epoch in 20 s → **Stop** → checkpoint
written, `stop_reason=stopped`, job status `stopped`.

## Registry

`data/registry.sqlite` (WAL), two tables so far:

- `models` — id, name, interval, architecture, window, status, label/feature
  config, checkpoint path + version, metrics.
- `training_runs` — one row per attempt: config, final metrics, per-class
  metrics, confusion matrix, epochs done, stop reason, duration, device.

Statuses in use: `training`, `candidate`, `draft`, `failed`. The plan's
`shadow` and `live` states are reserved for the execution phase and are not
written yet.

Checkpoints land in `data/checkpoints/<model_id>/v<n>.pt` via a temp file and
`os.replace`, so a crash mid-write cannot leave a half checkpoint. Each one
carries the state dict, the scaler, the feature names, the window, the label
order, and the metrics — enough to reconstruct inference without the registry.

## Not implemented here

- Compare view (side-by-side metrics, overlaid equity curves).
- Shadow / live promotion gates.
- `predictions_log` and the ADWIN drift loop.
- Walk-forward re-fitting across folds (current split is a single 70/15/15).
- Equity curve and Sharpe from test predictions.

Those are the remaining pieces of step 2 plus steps 7-8.
