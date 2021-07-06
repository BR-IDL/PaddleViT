

# training
## single-gpu
#CUDA_VISIBLE_DEVICES=5 python3  train.py --config ./configs/SETR/SETR_MLA_Large_480x480_80k_pascal_context_bs_8.yaml

## multi-gpu
#CUDA_VISIBLE_DEVICES=2,4,7 python3 -u -m paddle.distributed.launch train.py --config ./configs/SETR/SETR_MLA_Large_480x480_80k_pascal_context_bs_8.yaml

# testing
## single-gpu
#CUDA_VISIBLE_DEVICES=3 python3  val.py  --config ./configs/SETR/SETR_MLA_Large_480x480_80k_pascal_context_bs_8.yaml \
#    --model_path ./pretrain_models/setr/SETR_MLA_pascal_context_b8_80k.pdparams

## multi-gpu
CUDA_VISIBLE_DEVICES=2,3,4,5,7 python3 -u -m paddle.distributed.launch val.py  --config ./configs/SETR/SETR_MLA_Large_480x480_80k_pascal_context_bs_8.yaml \
    --model_path ./pretrain_models/setr/SETR_MLA_pascal_context_b8_80k.pdparams

