CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/mobilevit_xxs.yaml' \
-dataset='imagenet2012' \
-batch_size=128 \
-eval \
-data_path='/dataset/imagenet' \
-pretrained='output/train-20211027-22-10-32/MobileViT-Epoch-300-Loss-2.4265256190596847-EMA'
