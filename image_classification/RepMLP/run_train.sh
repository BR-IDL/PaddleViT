CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg='./configs/repmlpres50_light_224_train.yaml' \
-dataset='imagenet2012' \
-batch_size=4 \
-data_path='/dataset/imagenet' \
-amp
