# Multiple Node Training
English | [简体中文](./README_cn.md)

PaddleVit also supports multi-node distributed training under collective mode.

Here we provides a simple tutorial to modify multi-gpus training scripts 
to multi-nodes training scripts for any models in PaddleViT.

This folder takes ViT model as an example.

## Tutorial
For any models in PaddleViT, one can implement multi-node training by modifying 
`main_multi_gpu.py`.
1. Just add arguments `ips='[host ips]' ` in `dist.spawn()`.
2. Then run training script in every host.

## Training example: ViT
Suppose you have 2 hosts (denoted as node) with 4 gpus on each machine. 
Nodes IP addresses are `192.168.0.16` and `192.168.0.17`.

1. Then modify some lines of `run_train_multi_node.sh`:
    ```shell
    CUDA_VISIBLE_DEVICES=0,1,2,3 # number of gpus
    
    -ips= '192.168.0.16, 192.168.0.17' # seperated by comma
    ```
2. Run training script in every host:
    ```shell
    sh run_train_multi.sh
    ```

##Multi-nodes training with one host
It is possible to try multi-node training even when you have only one machine.

1. Install docker and paddle. For more details, please refer 
    [here](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/docker/fromdocker.html).

2. Create a network between docker containers.
    ```shell
    docker network create -d bridge paddle_net
    ```
3. Create multiple containers as virtual hosts/nodes. Suppose creating 2 containers 
with 2 gpus on each node.
    ```shell
    docker run --name paddle0 -it -d --gpus "device=0,1" --network paddle_net\
    paddlepaddle/paddle:2.2.0-gpu-cuda10.2-cudnn7 /bin/bash
    docker run --name paddle1 -it -d --gpus "device=2,3" --network paddle_net\
    paddlepaddle/paddle:2.2.0-gpu-cuda10.2-cudnn7 /bin/bash
    ```
    >   Noted: 
    >   1. One can assign one gpu device to different containers. But it may occur OOM since multiple models will run on the same gpu. 
    >   2. One should use `-v` to bind PaddleViT repository to container.

4. Modify `run_train_multi_node.sh` as described above and run the training script on every container.
   
    >   Noted: One can use `ping` or `ip -a` bash command to check containers' ip addresses. 

