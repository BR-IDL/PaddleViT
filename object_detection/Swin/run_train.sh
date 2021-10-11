CUDA_VISIBLE_DEVICES=7 \
python main_single_gpu.py \
-cfg='./configs/swin_s_maskrcnn.yaml' \
-dataset='coco' \
-batch_size=2 \
-data_path='/dataset/coco' \
-pretrained='./weights/mask_rcnn_swin_tiny_patch4_window7'
#-pretrained='./weights/swin_small_patch4_window7_224'

