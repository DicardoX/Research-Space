# Notes for Papers

[https://github.com/DicardoX/Notes_for_Papers](https://github.com/DicardoX/Notes_for_Papers)

> This repository is designed to record personal notes for reading papers.

-----

## Available Notes

### 1. Deployment of DNN Service

#### 1.1 Cluster-level & Co-location

- ***Nexus: A GPU Cluster Engine for Accelerating DNN-Based Video Analysis***

    - Abstract: GPU 集群层面的 DNN 服务调度，考虑混布，在满足低时延的同时实现高利用率
    - Link: [Notes for Nexus](https://github.com/DicardoX/Notes_for_Papers/tree/main/Nexus)

  

--------



### 2. Distributed Deep Learning

#### 2.1 Parameter Server

##### 2.1.1 第三代参数服务器架构

- ***Scaling Distributed Machine Learning with the Parameter Server***
    - Abstract: 第三代参数服务器架构，支持 scale 和 fault tolerance，异步通信，网络参数切分，灵活的一致性模型
    - Link: [Notes for Parameter Server (3rd)](https://github.com/DicardoX/Notes_for_Papers/tree/main/Parameter_Server_3rd)

#### 2.2 Ring All-reduce

##### 2.2.1 Ring All-reduce 工具

- ***Horovod: fast and easy distributed deep learning in TensorFlow***
    - Abstract: 基于 Baidu Ring Allreduce 框架进行代码实现和改进，python package，使用 Nvidia NCCL 内置的优化版本 ring allreduce，支持单服务器上的多 GPU 部署，部分 API 改进
    - Link: [Notes for Horovod)](https://github.com/DicardoX/Notes_for_Papers/tree/main/Horovod)

##### 2.2.2 分布式训练时 SGD 的 Large Minibatch 实现，可集成到 Ring All-reduce 方法

- ***Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour***
    - Abstract: Facebook 提出的分布式训练下 Distributed SGD 过程在应用大的 minibatch 的同时，保证训练准确性的方法，建立学习率与 batch size 的线性函数关系，每个 GPU/worker 的 bs 确定，通过调整 GPU 数目来改变整体的 minibatch size。由于将求梯度时的 average 操作限制在 per-worker 级别，可以集成到 ring allreduce 等仅支持加法操作的梯度聚合方法上。一点启发是，可以通过 profile 小的 minibatch，来评估大的 minibatch。
    - Link: [Notes for Large Minibatch Distributed SGD)](https://github.com/DicardoX/Notes_for_Papers/tree/main/Large_Minibatch_Distributed_SGD)