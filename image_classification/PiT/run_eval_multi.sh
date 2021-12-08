CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu.py \
    -cfg='./configs/pit_xs.yaml' \
    -dataset='imagenet2012' \
    -batch_size=64 \
    -data_path='/dataset/imagenet' \
    -eval \
    -pretrained='./pit_xs'
