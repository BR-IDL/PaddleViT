CUDA_VISIBLE_DEVICES=0,1 \
python main_multi_gpu.py \
-cfg='./configs/detr_resnet50.yaml' \
-dataset='coco' \
-batch_size=2 \
-data_path='/dataset/coco' \
-pretrained='./detr_resnet50'
