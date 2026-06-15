torchrun --standalone --nproc_per_node=1 generate_images.py \
	--outdir=out/drift-cifar-10-nocfg \
	--subdirs \
	--seeds=0-49999 \
	--preset=drift-cifar-10 \
	--max-batch-size=2048 \
	--encoder-batch-size=2048
