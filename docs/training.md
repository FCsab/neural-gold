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
the majority-class baseline, accuracy, expected realised return per predicted
class, the signal edge (mean return of longs minus mean return of stops), long
coverage, and the validation-loss curve.

Accuracy alone is never the promotion criterion: a model that always predicts
"timeout" can look respectable and have no edge. A run is marked **candidate**
only when it beats the majority baseline on balanced accuracy by more than
2 points; anything else lands in the registry as **draft**, with its metrics
attached so the result is inspectable rather than discarded.

## Measured on this machine

Two-minute budget, 210,322 rows, window 64, batch 512, 21 features, GPU:

```
device            cuda (AMD Radeon Graphics, probe passed)
epochs            10 in 121.2 s
balanced accuracy 0.403
baseline          0.485   <- majority class share
accuracy          0.417
signal edge       +0.006%
long coverage     45.6%
test rows         31,549
parameters        105,795
status            draft (did not beat baseline)
```

That is the honest result of two minutes on a near-random-walk target: the
plumbing works, the edge is not there yet. The registry keeps the model as a
draft with its metrics, rather than pretending it passed.

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
