CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg='./configs/styleformer_celeba.yaml' \
-dataset='celeba' \
-batch_size=32 \
-eval \
-pretrained='./celeba' \
-data_path='../img_align_celeba'
