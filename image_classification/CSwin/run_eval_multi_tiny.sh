CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu.py \
-cfg='./configs/cswin_tiny_224.yaml' \
-dataset='imagenet2012' \
-batch_size=128 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./output/my-train-20211012-21-08-50/cswin-Epoch-285-Loss-2.8287537443471553-EMA'
#-pretrained='./cswin_tiny_224' \
