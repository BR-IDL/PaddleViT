CUDA_VISIBLE_DEVICES=0 \
python pvtv2_det/object_detection/PVTv2/main_single_gpu.py \
-cfg='pvtv2_det/object_detection/PVTv2/configs/pvtv2_b0.yaml' \
-dataset='coco' \
-batch_size=1 \
-data_path='data/val' \
-eval \
-pretrained='/home/aistudio/data/data108821/retinanet_pvt_v2_b0_fpn_1x_coco'
