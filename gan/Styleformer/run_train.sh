CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg='./configs/styleformer_cifar10.yaml' \
-dataset='cifar10' \
-batch_size=32 \
-pretrained='./cifar10'
