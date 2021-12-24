CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu.py \
    -cfg='./configs/cvt-13-224x224.yaml' \
    -dataset='imagenet2012' \
    -batch_size=256 \
    -data_path='/dataset/imagenet' \
    -eval \
    -pretrained='./CvT-13-224x224-IN-1k'