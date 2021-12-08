CUDA_VISIBLE_DEVICES=0,1,2,3 \
#CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main_multi_gpu.py \
    -cfg='./configs/pit_xs.yaml' \
    -dataset='imagenet2012' \
    -batch_size=64 \
    -data_path='/dataset/imagenet' \
    -eval \
    -pretrained='./pit_xs'
