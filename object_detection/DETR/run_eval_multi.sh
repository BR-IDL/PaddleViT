CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu.py \
-cfg='./configs/detr_resnet50.yaml' \
-dataset='coco' \
-batch_size=4 \
-data_path='/dataset/coco' \
-eval \
-pretrained='./detr_resnet50'
