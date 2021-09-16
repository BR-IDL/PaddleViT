CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg='./configs/crossvit_base_224.yaml' \
-dataset='imagenet2012' \
-batch_size=128 \
-data_path='/home/fq/data/ILSVRC2012_img_val' \
-eval  \
-pretrained='./port_weights/pd_crossvit_base_224.pdparams'
