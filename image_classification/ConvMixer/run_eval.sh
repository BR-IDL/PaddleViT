CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg='./configs/convmixer_1536_20.yaml' \
-dataset='imagenet2012' \
-batch_size=32 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./convmixer_1536_20'
