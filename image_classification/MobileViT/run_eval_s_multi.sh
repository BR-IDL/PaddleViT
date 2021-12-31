CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/mobilevit_s.yaml' \
-dataset='imagenet2012' \
-batch_size=64 \
-eval \
-data_path='/dataset/imagenet' \
-pretrained='./output/train-20211103-21-27-27/MobileViT-Epoch-300-Loss-1.8248680274084845-EMA'
