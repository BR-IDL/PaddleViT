#!/bin/bash
cur_time=`date  +"%Y%m%d%H%M"`
job_name=ppvit_job${cur_time}
#group_name="idl-40g-0-yq01-k8s-gpu-a100-8"
group_name="idl-32g-1-yq01-k8s-gpu-v100-8"
job_version="paddle-fluid-custom"
image_addr="iregistry.baidu-int.com/idl/pytorch1.7.1:ubuntu16.04-cuda10.1_cudnn7_paddle_dev_0211"
start_cmd="sh run_pretrain_multi_vit_l_pdc.sh"
k8s_gpu_cards=8
wall_time="00:00:00"
k8s_priority="high"
file_dir="."


paddlecloud job --ak 560ffe9013d3592e8119e5c7e811e796 --sk 19b5ecd6b8305d81ae27d596a7f2fe22 \
        train --job-name ${job_name} \
        --job-conf config.ini \
        --group-name ${group_name} \
        --start-cmd "${start_cmd}" \
        --file-dir ${file_dir} \
        --job-version ${job_version}  \
        --k8s-gpu-cards ${k8s_gpu_cards} \
        --k8s-priority ${k8s_priority} \
        --wall-time ${wall_time} \
        --image-addr ${image_addr} \
        --is-standalone 1
