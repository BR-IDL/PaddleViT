CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
    -cfg='./configs/cvt-13-224x224.yaml' \
    -dataset='imagenet2012' \
    -batch_size=8 \
    -data_path='/dataset/imagenet' \
    -eval \
    -pretrained='./cvt_13_new'
