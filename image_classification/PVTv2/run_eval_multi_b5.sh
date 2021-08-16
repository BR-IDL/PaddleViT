CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/pvtv2_b5.yaml' \
-dataset='imagenet2012' \
-batch_size=32 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./pvtv2_b5' \
-ngpus=4
