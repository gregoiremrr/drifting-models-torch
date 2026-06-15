export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1

# Pretrain the ResNet-MAE feature encoder consumed by the drift loss.
torchrun --standalone --nproc_per_node=4 train_mae.py \
    --outdir=training-runs/cifar10-mae \
    --data=../datasets/cifar10.zip \
    --preset=mae-cifar10 \
    --max-batch-gpu=128 \
    --status=1Mi \
    --snapshot=8Mi \
    --checkpoint=32Mi
