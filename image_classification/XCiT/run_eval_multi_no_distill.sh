if [ $# -eq 1 ]
then
    cfg_file="./configs/$1.yaml"
    model_file="./$1.pdparams"
else
    cfg_file="./configs/xcit_large_24_p8_224.yaml"
    model_file="./xcit_large_24_p8_224.pdparams"
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu_no_distill.py \
-cfg=$cfg_file \
-dataset='imagenet2012' \
-batch_size=128 \
-data_path='/dataset/imagenet' \
-pretrained=$model_file \
-eval \
-amp
