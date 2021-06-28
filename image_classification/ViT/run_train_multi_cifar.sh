#CUDA_VISIBLE_DEVICES=4,5,6,7 python main_multi_gpu.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_multi_gpu.py \
-dataset="cifar10" \
-batch_size=64 \
-image_size=224 \
