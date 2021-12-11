#CUDA_VISIBLE_DEVICES=0,1,2,3 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/swin_tiny_patch4_window7_224.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./output/train-20211006-19-17-58/swin-Epoch-298-Loss-3.0057902509114243' \
#-pretrained='./output/train-20210929-21-17-57/swin-Epoch-286-Loss-3.018214573891194' \
#-pretrained='./output/train-20210929-21-17-57/swin-Epoch-298-Loss-3.021707329043735' \
#-pretrained='./output/train-20210929-21-17-57/swin-Epoch-150-Loss-3.256427281403651' \
#-pretrained='./output/train-20210929-21-17-57/swin-Epoch-128-Loss-3.30339895277557' \
