CUDA_VISIBLE_DEVICES=1 \
python main_single_gpu.py \
-cfg='./configs/detr_resnet50.yaml' \
-dataset='coco' \
-batch_size=2 \
-data_path='/dataset/coco' \
-pretrained='./detr_resnet50'
