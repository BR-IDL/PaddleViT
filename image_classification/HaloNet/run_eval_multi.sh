CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu.py \
    -cfg='./configs/halonet_50ts_256.yaml' \
    -dataset='imagenet2012' \
    -batch_size=256 \
    -data_path='/dataset/imagenet' \
    -eval \
    -pretrained='./halonet_50ts_256'
