CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu.py \
    -cfg='./configs/poolformer_m48.yaml' \
    -dataset='imagenet2012' \
    -batch_size=128 \
    -data_path='/dataset/imagenet' \
    -eval \
    -pretrained='./poolformer_m48'
