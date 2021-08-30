CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu.py \
-cfg='./configs/styleformer_cifar10.yaml' \
-dataset='cifar10' \
-batch_size=32 \
#-pretrained='./cifar10'
