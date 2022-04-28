GLOG_v=0 python3 -m paddle.distributed.launch --gpus="4,5,6,7" \
main_multi_gpu.py \
-cfg="./swin_small_patch4_window7_224.yaml" \
-dataset="ABAW" \
-batch_size=256 \
-data_folder="../abaw_dataset/aligned_MTCNNXue/" \
-anno_folder="../abaw_dataset/Third_ABAW_Annotations/EXPR_Classification_Challenge/" \
-pretrained="./ABAW3_SwinS_Negative_Iter20k.pdparams" \
-class_type="negative" \
-eval \
-amp \
