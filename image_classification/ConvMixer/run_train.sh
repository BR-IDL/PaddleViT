CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg='./configs/convmixer_1024_20.yaml' \
-dataset='imagenet2012' \
-batch_size=4 \
-data_path='/dataset/imagenet' \
