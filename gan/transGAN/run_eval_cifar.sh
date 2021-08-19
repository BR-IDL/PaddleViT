CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg="./configs/transgan_cifar10.yaml" \
-dataset='cifar10' \
-batch_size=64 \
-eval \
-pretrained='./transgan_cifar10'
