CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/mobilevit_xxs.yaml' \
-dataset='imagenet2012' \
-batch_size=128 \
-data_path='/dataset/imagenet' \
-resume='output/train-20211026-20-05-16/MobileViT-Epoch-109-Loss-2.623020797677231' \
-last_epoch=109 \
-amp
