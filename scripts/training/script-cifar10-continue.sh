export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1

# Resume drifting-generator training: point --outdir at the existing run dir
# (the one containing training-state-*.pt).
torchrun --standalone --nproc_per_node=4 train.py \
    --outdir=training-runs/cifar10/<run-dir> \
    --data=datasets/cifar10.zip \
    --mae-pkl=training-runs/cifar10-mae/<run-dir>/model-snapshot-<latest>.pkl \
    --preset=drift-cifar10 \
    --no-fp16 \
    --status=1Mi \
    --snapshot=8Mi \
    --checkpoint=32Mi
