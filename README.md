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

##### 2.1.2 面向对分布式训练任务收敛时间及速度的在线预测

- ***Optimus: An Efficient Dynamic Resource Scheduler for Deep Learning Clusters***

    - Abstract: Optimus 是一个面向分布式训练 job 的调度器，主要目标是最小化 JCT，主要分为对 DL 模型收敛时间的预测，对模型训练速度的评估，以及一个考虑资源分配和 task（相较于 job 的更小级别）放置三个部分。Optimus 对 DL 模型收敛时间的预测基于 SGD 收敛速率为 O (1 / k) 的事实（该结论对学习率的变化有要求），使用 non-negative least squares (NNLS) 求解器来进行在线的拟合。为了建立性能评估模型，Optimus 建立了数学化分布式训练任务和资源的系统模型，并分析了包括 worker 前向（与 bs 相关） & 后向（与 bs 无关）时间，数据传输时间（部分模型大小 / 带宽），parameter server (ps) 的参数更新时间，以及通信开销（与 worker 和 ps 的数目成线性关系）在内的总时间 T。基于上述分析，Optimus 构造了一个表示训练速度 f = 1 / T 的函数，注意该函数以 worker 和 ps 的数目为自变量，对于同步和异步训练有些许区别（前者需要进一步考虑用户对 overall batch size M 的输入），并以 NNLS 拟合的方式，通过预训练和在线收集训练数据建立和不断改进模型。最后，基于任务剩余的代数以及上面得到的训练速度函数 f，Optimus 定义了边际收益，来以启发式（贪心）的方式进行资源分配（不用整数规划是因为非线性甚至非凸，且 NP-hard）。对于 task 放置（实际上是对 job 所被分配的 workers 和 ps 进行 servers 资源的分配，分配完后均衡地放在这些 servers 上），Optimus 分析了达到最小化最大数据传输的方式，包括使用最少的 servers 来部署以及均衡地在这些 servers 上面放置 workers 和 ps，并以此提出了基于贪心的分布式策略。此外，为了解决 stragglers 的问题，Optimus 分别对 workers 的训练速度进行均衡，并提出了对 ps 负载进行均衡的方法。

        然而，Optimus 对模型训练时间的预测在其他工作中被证明为过分简化了损失函数的变化曲线，在实际集群中并不总是适用的。

    - Link: [Notes for Optimus](https://github.com/DicardoX/Notes_for_Papers/tree/main/Optimus)

##### 2.1.3 面向无信息和部分信息场景的离散优先级抢占式调度算法及合并放置约束的深入讨论

- ***Tiresias: A GPU Cluster Manager for Distributed Deep Learning***

    - Abstract: Tiresias 是一个面向参数服务器（Parameter Server）架构下分布式训练 job 的调度器，主要包括一个 2DAS 调度器，以及基于模型结构对合并放置约束的放宽策略。首先，Tiresias 讨论了同时考虑时间（JCT）和空间（资源分配）两个维度（2D）以计算优先级的重要性，以及优先级离散化的优势。Tiresias 进行架构设计的前提条件包括：DL job 的执行时间通常是不可预测的，job 的特点无法提前获知，以及同步 PS 架构下 DL job 的资源分配具有 all-or-nothing 的特点。

        - 对于 2DAS 调度器：当无先验信息时，优先级函数采用 2D-LAS 算法，job 获取的资源量等于当前执行时间 * 分配资源量；若提供部分先验信息（比如执行几轮后，但 paper 中并没有说明如何通过几轮的信息拟合得到 JCT 分布，只说需要管理者提供该分布），优先级的值则与 Gittins index value 相等，Gittins index 值代表该 job 获取一定量服务后在拿到下一个服务量后能完成的可能性，值越高表示优先级越高。
        - 对于优先级离散化：参考多级反馈队列（MLFQ）的架构进行设计。LAS时，对相同队列里的任务采用基于 start time 的 FIFO；Gittins index 中，service quantum 被设置为当前队列的服务量上界，当 job 消耗完 \Delta 时被降级。相同队列中的 jobs 以各自的 Gittins index 值来调度，最低队列（\Delta 为无穷，无法计算 Gittins index value）以 FIFO 的方式。
        - 对于合并放置约束：Tiresias 使用模型结构的倾斜程度来预测 jobs 是否对合并放置敏感。部分模型的某些层是很大的 tensors，而在模型聚合时该层参数的信息大小与 tensors 大小相关。因此，聚合大 tensors 更容易受到网络竞争的影响。由于每个 parameter server 都会周期性地向每个 worker 发出自己拥有的更新后的模型参数，Tiresias 通过监控网络通信来获取模型倾斜信息，并构造了一个自己的监控工具。

        整篇 paper 最 fancy 的地方在于面向无 JCT 分布信息和部分信息抢占式 2D 调度算法设计，参考多级反馈队列设计的优先级离散化架构，以及基于网络通信对模型 tensors 倾斜信息的监控方法。

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

##### 2.3.1 PS 和 Ring All-reduce 算法的统一架构

- ***(BytePS) A Unified Architecture for Accelerating Distributed DNN Training in Heterogeneous GPU/CPU Clusters***
    - Abstract: ByteDance 提出的将 PS 和 Ring All-reduce 两种架构综合考虑的一种统一集群内通信架构，利用集群中空闲的 CPU 和带宽资源，并将 PS 和 Ring All-reduce 成功表述为统一架构下的特殊情况。SS（Summation Service）只运行在 CPU 中（包括 CPU 机器和 GPU 机器），负责从 CS 侧接收 tensors，把 tensors 加起来，再返回给 CS；CS (Communication Service) 只运行在 GPU 中，负责同步多个局部 GPU 之间的 tensors。该架构能够根据集群中可用 CPU 和 GPU 的相对数目，通过硬件 profile 动态决定 SS 中分配在 CPU / GPU 的数据的比例，这样得到的机器间通信是延迟最优的。同时，BytePS 研究了机器内拓扑结构对通信效率的影响，针对 PCIe-only 拓扑提出了 CPU-assisted aggregation 策略（可 pipeline），针对 NVLink-based 拓扑使用 reduce and broadcast 策略。BytePS 使 CPU 仅负责 sum（SS），而 GPU 负责 FP、BP 和 parameter update。由于在 sum 之前 GPU 就要 update 参数，因此破坏了 PS 原有对异步并行的支持性， BytePS 因此提出支持异步的参数更新算法，即向 CPU 传输 delta w，并证明了其和 PS 异步并行的等效性。
    - Link: [Notes for BytePS](https://github.com/DicardoX/Notes_for_Papers/tree/main/BytePS)

#### 2.4 Co-optimizing at job-level and cluster-level

##### 2.4.1 Job-level (bs, lr) 和 cluster-level (resource allocation) 的 modeling 和 co-optimization

- ***Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning***

    - Abstract: Petuum 提出的将分布式训练中的 metric modeling 和 resource scheduling 结合起来 co-optimize 的分布式训练和调度架构。Pollux 可以同时用在 parameter server 和 all-reduce 两类分布式架构，原因是重点关注 goodput 为主的性能指标，而非 ps 和 all-reduce 等架构中最重要的通信量的优化，这也是 Pollux 可以优化（考虑不足）的一个点。考虑的主要 metrics 为 statistical efficiency（表征每一轮迭代模型精度能够进步多少） 和 throughput，并分别对这两个 metric 的 estimate 进行了 modeling。Pollux 使用 gradient accumulation 来进一步扩大 batch size，经过 s 轮的 GPU 本地梯度聚合后再进行全局的梯度聚合。基于这两个 metric，Pollux 提出了自己的调度和优化 metric，即每个job 的 goodput。整个 Pollux 的架构分为两层：

        -  job 层面，PolluxAgent 动态调整 bs 和 lr 以更好地利用资源，收集 bs 和 $T_{iter}$ 等信息，基于信息拟合 efficiency 和 throughput 的函数，进而获取每个 job 的 goodput 函数。最后，周期性地向 PolluxSched 报告 goodput 函数，并等待新一轮资源分配后再调整 bs 和 lr
        - Cluster 层面：PolluxSched 基于 jobs 的 goodput 动态重分配资源，通过最大化 fitness 函数来获取理论最优的分配，并考虑多个集群层面的目标，包括 fairness，goodput，reallocataion overhead，inference slowdown 等

        整篇 paper 最 fancy 的地方在于它的建模过程，包括 metric modeling 和 scheduling (optimization) modeling。

    - Link: [Notes for Pollux](https://github.com/DicardoX/Notes_for_Papers/tree/main/Pollux)

