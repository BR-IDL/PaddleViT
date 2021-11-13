CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
    -cfg='./configs/pit_ti.yaml' \
    -dataset='imagenet2012' \
    -batch_size=16 \
    -data_path='/dataset/imagenet' \
    -eval \
    -pretrained='./pit_ti'
