CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/resmlp_12_224.yaml' \
-dataset='imagenet2012' \
-batch_size=16 \
-data_path='/dataset/imagenet' \
-pretrained='./resmlp_12_distilled_224.pdparams' \
-eval \
-amp
