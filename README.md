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
    - Link: [Notes for Horovod](https://github.com/DicardoX/Notes_for_Papers/tree/main/Horovod)

##### 2.2.2 分布式训练时 SGD 的 Large Minibatch 实现，可集成到 Ring All-reduce 方法

- ***Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour***
    - Abstract: Facebook 提出的分布式训练下 Distributed SGD 过程在应用大的 minibatch 的同时，保证训练准确性的方法，建立学习率与 batch size 的线性函数关系，每个 GPU/worker 的 bs 确定，通过调整 GPU 数目来改变整体的 minibatch size。由于将求梯度时的 average 操作限制在 per-worker 级别，可以集成到 ring allreduce 等仅支持加法操作的梯度聚合方法上。一点启发是，可以通过 profile 小的 minibatch，来评估大的 minibatch。
    - Link: [Notes for Large Minibatch Distributed SGD](https://github.com/DicardoX/Notes_for_Papers/tree/main/Large_Minibatch_Distributed_SGD)

#### 2.3 Combination of PS and Ring All-reduce

##### 2.3.1 BytePS：PS 和 Ring All-reduce 算法的统一架构

- ***A Unified Architecture for Accelerating Distributed DNN Training in Heterogeneous GPU/CPU Clusters***
    - Abstract: ByteDance 提出的将 PS 和 Ring All-reduce 两种架构综合考虑的一种统一集群内通信架构，利用集群中空闲的 CPU 和带宽资源，并将 PS 和 Ring All-reduce 成功表述为统一架构下的特殊情况。SS（Summation Service）只运行在 CPU 中（包括 CPU 机器和 GPU 机器），负责从 CS 侧接收 tensors，把 tensors 加起来，再返回给 CS；CS (Communication Service) 只运行在 GPU 中，负责同步多个局部 GPU 之间的 tensors。该架构能够根据集群中可用 CPU 和 GPU 的相对数目，通过硬件 profile 动态决定 SS 中分配在 CPU / GPU 的数据的比例，这样得到的机器间通信是延迟最优的。同时，BytePS 研究了机器内拓扑结构对通信效率的影响，针对 PCIe-only 拓扑提出了 CPU-assisted aggregation 策略（可 pipeline），针对 NVLink-based 拓扑使用 reduce and broadcast 策略。BytePS 使 CPU 仅负责 sum（SS），而 GPU 负责 FP、BP 和 parameter update。由于在 sum 之前 GPU 就要 update 参数，因此破坏了 PS 原有对异步并行的支持性， BytePS 因此提出支持异步的参数更新算法，即向 CPU 传输 delta w，并证明了其和 PS 异步并行的等效性。
    - Link: [Notes for BytePS](https://github.com/DicardoX/Notes_for_Papers/tree/main/BytePS)

#### 2.4 Co-optimizing at job-level (bs, lr) and cluster-level (resource allocation)

##### 2.4.1 Pollux：job-level 和 cluster-level 的 modeling 和 co-optimization

- ***Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning***

    - Abstract: Petuum 提出的将分布式训练中的 metric modeling 和 resource scheduling 结合起来 co-optimize 的分布式训练和调度架构。Pollux 考虑的主要 metrics 为 statistical efficiency（表征每一轮迭代模型精度能够进步多少） 和 throughput，并分别对这两个 metric 的 estimate 进行了 modeling。基于这两个 metric，Pollux 提出了自己的调度和优化 metric，即每个job 的 goodput。整个 Pollux 的架构分为两层：

        -  job 层面，PolluxAgent 动态调整 bs 和 lr 以更好地利用资源，收集 bs 和 $T_{iter}$ 等信息，基于信息拟合 efficiency 和 throughput 的函数，进而获取每个 job 的 goodput 函数。最后，周期性地向 PolluxSched 报告 goodput 函数，并等待新一轮资源分配后再调整 bs 和 lr
        - Cluster 层面：PolluxSched 基于 jobs 的 goodput 动态重分配资源，通过最大化 fitness 函数来获取理论最优的分配，并考虑多个集群层面的目标，包括 fairness，goodput，reallocataion overhead，inference slowdown 等

        整篇 paper 最 fancy 的地方在于它的建模过程，包括 metric modeling 和 scheduling (optimization) modeling。

    - Link: [Notes for Pollux](https://github.com/DicardoX/Notes_for_Papers/tree/main/Pollux)

