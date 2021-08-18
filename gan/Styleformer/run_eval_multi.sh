CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/styleformer_cifar10.yaml' \
-dataset='cifar10' \
-batch_size=64 \
-eval \
-pretrained='./cifar10'
