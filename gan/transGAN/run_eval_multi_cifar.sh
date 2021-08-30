CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu.py \
-cfg="./configs/transgan_cifar10.yaml" \
-dataset='cifar10' \
-batch_size=64 \
-eval \
-pretrained='./transgan_cifar10'
