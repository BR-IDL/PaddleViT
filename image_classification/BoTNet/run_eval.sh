CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg='./configs/botnet50_224.yaml' \
-dataset='imagenet2012' \
-batch_size=4 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./botnet50' \
