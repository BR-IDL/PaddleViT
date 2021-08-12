CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg="transgan_cifar10.yaml" \
-dataset='cifar10' \
-batch_size=32 \
-pretrained='./transgan_cifar10'