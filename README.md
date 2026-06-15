# Drifting Models (PyTorch)

A multi-GPU PyTorch port of **"Generative Modeling via Drifting"**
(arXiv:2602.04770), adapted to **CIFAR-10 in pixel space (no VAE)**.

A drifting model is a *one-step* (1-NFE) generator: a LightningDiT transformer
maps Gaussian noise directly to an image in a single forward pass. There is no
iterative sampler. Training does not use a fixed regression target; instead a
**drifting field** — an anti-symmetric attraction/repulsion mean-shift field
estimated within each minibatch via a doubly-normalized softmax kernel — pushes
the generated samples toward the data distribution. The drift is computed in the
feature space of a frozen, pretrained **ResNet-MAE**, so you train the MAE first
and the generator second.

The surrounding infrastructure (`dnnlib`, `torch_utils`, post-hoc EMA,
`dataset_tool.py`, FID/FD-DINOv2 metrics, persistence-based pickling) is borrowed
from NVIDIA's [EDM2](https://github.com/NVlabs/edm2). The drifting model, MAE,
drift loss, memory bank, samplers, and configuration layout sit on top.

## Layout

```
.
├── train.py                    # Build a config and launch generator (drift) training.
├── train_mae.py                # Build a config and launch ResNet-MAE pretraining.
├── generate_images.py          # Sample (1-NFE) from a saved snapshot pickle.
├── calculate_metrics.py        # FID and FD-DINOv2 against a reference dataset.
├── reconstruct_phema.py        # Post-hoc EMA reconstruction (EDM2).
├── dataset_tool.py             # Pack a folder of images into a zip dataset.
├── training/
│   ├── training_loop.py        # Generator (drift) training loop.
│   ├── training_loop_mae.py    # ResNet-MAE pretraining loop.
│   ├── model.py                # DriftingModel wrapper + 1-NFE sample().
│   ├── loss.py                 # DriftLoss (multi-feature, CFG negatives).
│   ├── drift_field.py          # Drifting field + per-feature drift loss.
│   ├── memory_bank.py          # Class-wise MoCo-style sample queues.
│   ├── networks_dit.py         # LightningDiT generator (SwiGLU/RoPE/RMSNorm/QK-norm/adaLN-zero).
│   ├── networks_mae.py         # ResNet-MAE encoder + U-Net decoder + get_activations().
│   ├── encoders.py             # StandardRGB ([-1,1] pixels) and Stability VAE encoders.
│   ├── schedulers.py           # LR schedules (incl. warmup_const_lr).
│   ├── phema.py                # Power-function and traditional EMA.
│   ├── monitoring.py           # W&B logging helpers.
│   ├── evaluation.py           # Distributed FID/MIND computation.
│   └── dataset.py              # Streaming image dataset (zip or folder).
├── torch_utils/                # Distributed, persistence, training stats (EDM).
├── dnnlib/                     # EasyDict, class/func construction by name (EDM).
├── scripts/                    # Shell scripts: env setup, training, metrics.
├── datasets/                   # Place your packed datasets here.
├── training-runs/              # Output runs (one timestamped subdir per launch).
├── fid-refs/                   # Reference statistics for FID/FD-DINOv2.
└── out/                        # Generated images.
```

## Setup

```bash
# Install the Python environment (CUDA 12.x wheel; adjust as needed).
bash scripts/module.sh
```

A `Dockerfile` is also provided.

## End-to-end workflow

### 1. Pack the dataset

```bash
python dataset_tool.py convert \
    --source=raw_cifar/ \
    --dest=datasets/cifar10.zip \
    --resolution=32x32
```

Drifting models are class-conditional, so the dataset **must** have labels.

### 2. Pretrain the ResNet-MAE feature encoder

```bash
torchrun --standalone --nproc_per_node=4 train_mae.py \
    --outdir=training-runs/cifar10-mae \
    --data=datasets/cifar10.zip \
    --preset=mae-cifar10
```

This produces `model-snapshot-*.pkl` files containing the EMA encoder. The MAE
reconstructs randomly (2x2-patch) masked inputs and (over the final images)
fine-tunes a small linear classifier head.

### 3. Train the generator with the drift loss

```bash
torchrun --standalone --nproc_per_node=4 train.py \
    --outdir=training-runs/cifar10 \
    --data=datasets/cifar10.zip \
    --mae-pkl=training-runs/cifar10-mae/<run-dir>/model-snapshot-<latest>.pkl \
    --preset=drift-cifar10
```

Each launch creates a timestamped subdirectory inside `--outdir`. Pointing
`--outdir` at an existing run that contains a `training-state-*.pt` resumes from
the latest checkpoint. Use the `*-debug` presets (`mae-cifar10-debug`,
`drift-cifar10-debug`) for fast single-GPU smoke tests.

### 4. Generate images (1-NFE)

```bash
python generate_images.py \
    --model=training-runs/cifar10/<run-dir>/model-snapshot-<latest>-0.100.pkl \
    --sampler-fn=training.model.sample \
    --outdir=out --seeds=0-63 --guidance=1.0
```

`guidance` is the training-time-style CFG scale fed to the generator as a
conditioning input; `--n-sampling-steps` is accepted but ignored (the model is
one-step).

### 5. Compute reference statistics and FID

```bash
bash scripts/metrics/ref50k.sh      # build fid-refs/cifar10.pkl
bash scripts/metrics/fid50k.sh      # FID / FD-DINOv2
```

FID can also be computed inline during generator training via `--metrics` /
`--metric-ref`.

## How the drift training step works

Each generator step (`training/training_loop.py`):

1. **Fill the queues.** Freshly loaded reals are pushed into per-class
   (positive) and a global (unconditional) ring buffer (`memory_bank.py`,
   Appendix A.8) — a MoCo-style queue rather than a bespoke data loader.
2. **Draw a batch.** Sample `labels_per_step` class labels; draw positives and
   unconditional negatives from the queues.
3. **Generate + featurize.** Sample a CFG scale per label, generate
   `gen_per_label` samples per label, and extract multi-scale / multi-location
   features for reals and fakes through the frozen MAE (`get_activations`).
4. **Drift.** For each feature compute the drifting field (`drift_field.py`):
   normalize features, build the doubly-normalized softmax affinity over
   `[generated | negatives | positives]`, aggregate across temperatures
   (`R_list`), and regress the generator output onto its frozen drifted target.
   Classifier-free guidance (Appendix A.7) re-weights the unconditional reals as
   extra negatives by `(alpha - 1)(g - 1)/n_uncond`.

## Configuration

Both launchers split configuration into two preset dictionaries:

- `dataset_presets` holds everything intrinsic to the data: the network
  architecture (`net_kwargs`), the monitoring sampler (`sampler_kwargs`), EMA
  profiles, and the LR scheduler family (`lr_scheduler_kwargs`).
- `config_presets` describes a particular run: sample allocation
  (`labels_per_step`, `gen_per_label`, `pos_per_sample`, `neg_per_sample`),
  queue sizes, CFG range (`cfg_min`/`cfg_max`/`neg_cfg_pw`), drift temperatures
  (`R_list`), optimization budget, LR, warmup, and gradient clipping.

The two dictionaries must have disjoint keys (asserted at startup). Per-run
overrides are exposed as CLI flags.

## Monitoring

Loss, learning rate, gradient norm, gradient-clip coefficient, mean CFG scale,
and timing counters are pushed to W&B at every `--status` interval, alongside a
grid of 1-NFE samples from the EMA generator. The MAE loop logs reconstruction
loss, classification loss/accuracy, and `lambda_cls`.

## Credits

Drifting model method: "Generative Modeling via Drifting" (arXiv:2602.04770)
and its JAX reference implementation. Infrastructure built on NVIDIA's
[EDM](https://github.com/NVlabs/edm) and [EDM2](https://github.com/NVlabs/edm2).
