CUDA_VISIBLE_DEVICES=1 \
python main_single_gpu.py \
-cfg='./configs/styleformer_cifar10.yaml' \
-dataset='cifar10' \
-batch_size=64 \
-eval \
-pretrained='./cifar10'
