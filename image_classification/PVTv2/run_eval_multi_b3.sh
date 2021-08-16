CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu.py \
-cfg='./configs/pvtv2_b3.yaml' \
-dataset='imagenet2012' \
-batch_size=64 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./pvtv2_b3' \
-ngpus=4
