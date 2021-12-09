#多机多卡分布式训练

简体中文 | [English](./README.md)

PaddleViT 同样支持Collective多机多卡分布式训练。

##教程
对于每一个模型，用户可以直接通过修改对应模型文件夹下的`main_multi_gpu.py`
以进行多机训练。
1. 在`dist.spawn()`里加入`ips='[host ips]' `。
2. 在每个主机上运行代码。

##样例：ViT
这个文件夹提供了分布式训练ViT模型的代码和shell脚本。
假设你有2台主机，每个主机上有4张显卡。主机的ip地址为`192.168.0.16`和`192.168.0.17`。

1. 修改shell脚本`run_train_multi_node.sh`的参数
   ```shell
    CUDA_VISIBLE_DEVICES=0,1,2,3 # number of gpus
    
    -ips= '192.168.0.16, 192.168.0.17' # seperated by comma
    ```
2. 在每个主机上运行脚本代码。
    ```shell
    sh run_train_multi.sh
    ```
##单机上运行分布式训练
如果仅有一台主机，同样可以通过docker实现单机上的分布式训练。
1. 安装docker和paddle。[这里](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/docker/fromdocker.html)
可以下载paddlepaddle提供的docker镜像。
2. 创建docker容器间网络。
   ```shell
    docker network create -d bridge paddle_net
    ```
3. 创建多个docker容器作为虚拟主机，假设我们创建2个容器，并分别分配2个GPU。
    ```shell
    docker run --name paddle0 -it -d --gpus "device=0,1" --network paddle_net\
    paddlepaddle/paddle:2.2.0-gpu-cuda10.2-cudnn7 /bin/bash
    docker run --name paddle1 -it -d --gpus "device=2,3" --network paddle_net\
    paddlepaddle/paddle:2.2.0-gpu-cuda10.2-cudnn7 /bin/bash
    ```
    >   注意: 
    >   1. 可以将同一个GPU同时分配给多个容器，但是这可能会产生OOM错误，因为多个模型将同时运行在这个GPU上。 
    >   2. 使用`-v`挂载PaddleViT所在的目录。

