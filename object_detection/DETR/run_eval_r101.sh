CUDA_VISIBLE_DEVICES=7 \
python main_single_gpu.py \
-cfg='./configs/detr_resnet101.yaml' \
-dataset='coco' \
-batch_size=2 \
-data_path='/dataset/coco' \
-eval \
-pretrained='./detr_resnet101'
