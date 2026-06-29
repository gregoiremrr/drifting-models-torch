# Quick single-GPU smoke test of the drifting-generator config (no real run).
torchrun --standalone --nproc_per_node=1 train.py \
    --outdir=training-runs/cifar10 \
    --data=datasets/cifar10.zip \
    --mae-pkl=training-runs/cifar10-mae/<run-dir>/model-snapshot-<latest>.pkl \
    --preset=drift-cifar10-debug \
    --status=2 \
    --snapshot=20 \
    --checkpoint=40 \
    --dry-run
