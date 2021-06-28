CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/detr_resnet50.yaml' \
-dataset='coco' \
-batch_size=8 \
-data_path='/dataset/coco' \
-eval \
-pretrained='./detr_resnet50'
