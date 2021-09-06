CUDA_VISIBLE_DEVICES=1 \
python main_single_gpu.py \
-cfg='./configs/swin_t_maskrcnn.yaml' \
-dataset='coco' \
-batch_size=8 \
-data_path='/dataset/coco' \
-eval \
-pretrained='./weights/mask_rcnn_swin_tiny_patch4_window7_1x'
