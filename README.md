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

- ***Scaling Distributed Machine Learning with the Parameter Server***
    - Abstract: 第三代参数服务器架构，支持 scale 和 fault tolerance，异步通信，网络参数切分，灵活的一致性模型
    - Link: [Notes for Parameter Server (3rd)](https://github.com/DicardoX/Notes_for_Papers/tree/main/Parameter_Server_3rd)

#### 2.2 Ring All-reduce

- ***Horovod: fast and easy distributed deep learning in TensorFlow***
    - Abstract: 基于 Baidu Ring Allreduce 框架进行代码实现和改进，python package，使用 Nvidia NCCL 内置的优化版本 ring allreduce，支持单服务器上的多 GPU 部署，部分 API 改进
    - Link: [Notes for Horovod)](https://github.com/DicardoX/Notes_for_Papers/tree/main/Horovod)