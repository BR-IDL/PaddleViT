CUDA_VISIBLE_DEVICES=7 \
python main_single_gpu.py \
-cfg='./configs/swin_t_maskrcnn.yaml' \
-dataset='coco' \
-batch_size=2 \
-data_path='/dataset/coco' \
-pretrained='./weights/mask_rcnn_swin_tiny_patch4_window7'
