CUDA_VISIBLE_DEVICES=3 \
python main_single_gpu.py \
-cfg='./configs/swin_s_maskrcnn.yaml' \
-dataset='coco' \
-batch_size=8 \
-data_path='/dataset/coco' \
-eval \
-pretrained='./weights/mask_rcnn_swin_small_patch4_window7'
