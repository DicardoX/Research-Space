# Notes for Papers

[https://github.com/DicardoX/Notes_for_Papers](https://github.com/DicardoX/Notes_for_Papers)

> This repository is designed to record personal notes for reading papers.

-----



## 1. Deployment of DNN Service

### 1.1 Cluster-level & Co-location

- ***Nexus: A GPU Cluster Engine for Accelerating DNN-Based Video Analysis***
    - Abstract: GPU 集群层面的 DNN 服务调度，考虑混布，在满足低时延的同时实现高利用率
    
    - Link: [Note for Nexus](https://github.com/DicardoX/Notes_for_Papers/tree/main/Nexus)

--------



## 2. Distributed Deep Learning

### 2.1 Data Parallelism: Parameter Server

#### 2.1.1 第三代参数服务器架构

- ***Scaling Distributed Machine Learning with the Parameter Server***
    - Abstract: 第三代参数服务器架构，支持 scale 和 fault tolerance，异步通信，网络参数切分，灵活的一致性模型
    
    - Link: [Note for Parameter Server (3rd)](https://github.com/DicardoX/Notes_for_Papers/tree/main/Parameter_Server_3rd)

#### 2.1.2 面向对分布式训练任务收敛时间及速度的在线预测

- ***Optimus: An Efficient Dynamic Resource Scheduler for Deep Learning Clusters***

    - Abstract: Optimus 是一个面向分布式训练 job 的调度器，主要目标是最小化 JCT，主要分为对 DL 模型收敛时间的预测，对模型训练速度的评估，以及一个考虑资源分配和 task（相较于 job 的更小级别）放置三个部分。

    - Link: [Note for Optimus](https://github.com/DicardoX/Notes_for_Papers/tree/main/Optimus)

#### 2.1.3 面向无信息和部分信息场景的离散优先级抢占式调度算法及合并放置约束的深入讨论

- ***Tiresias: A GPU Cluster Manager for Distributed Deep Learning***
    - Abstract: Tiresias 是一个面向参数服务器（Parameter Server）架构下分布式训练 job 的调度器，主要包括一个 2DAS 调度器，以及基于模型结构对合并放置约束的放宽策略。
    
    - Link: [Note for Tiresias](https://github.com/DicardoX/Notes_for_Papers/tree/main/Tiresias)

--------



### 2.2 Data Parallelism: Ring All-reduce

#### 2.2.1 通信量优化的 Ring All-reduce 工具

- ***Horovod: fast and easy distributed deep learning in TensorFlow***
    - Abstract: 基于 Baidu Ring Allreduce 框架进行代码实现和改进，python package，使用 Nvidia NCCL 内置的优化版本 ring allreduce，支持单模型在单服务器上的多 GPU 部署，部分 API 改进
    
    - Link: [Note for Horovod](https://github.com/DicardoX/Notes_for_Papers/tree/main/Horovod)

#### 2.2.2 分布式训练时 SGD 的 Large Minibatch 实现，可集成到 Ring All-reduce 方法

- ***Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour***
    - Abstract: Facebook 提出的分布式训练下 Distributed SGD 过程在应用大的 minibatch 的同时，保证训练准确性的方法。
    
    - Link: [Note for Large Minibatch Distributed SGD](https://github.com/DicardoX/Notes_for_Papers/tree/main/Large_Minibatch_Distributed_SGD)

----------



### 2.3 Data Parallelism: Combination of PS and Ring All-reduce

#### 2.3.1 PS 和 Ring All-reduce 算法的统一架构

- ***(BytePS) A Unified Architecture for Accelerating Distributed DNN Training in Heterogeneous GPU/CPU Clusters***
    - Abstract: ByteDance 提出的将 PS 和 Ring All-reduce 两种架构综合考虑的一种统一集群内通信架构，利用集群中空闲的 CPU 和带宽资源，并将 PS 和 Ring All-reduce 成功表述为统一架构下的特殊情况。
    
    - Link: [Note for BytePS](https://github.com/DicardoX/Notes_for_Papers/tree/main/BytePS)

--------



### 2.4 Data Parallelism: Co-optimizing at Job-level and Cluster-level

#### 2.4.1 Job-level (bs, lr) 和 cluster-level (resource allocation) 的 modeling 和 co-optimization

- ***Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning***
    - Abstract: Petuum 提出的将分布式训练中的 metric modeling 和 resource scheduling 结合起来 co-optimize 的分布式训练和调度架构。
    
    - Link: [Note for Pollux](https://github.com/DicardoX/Notes_for_Papers/tree/main/Pollux)

---------



### 2.5 Model Parallelism: The Foundation of Model Parallelism

#### 2.5.1 局部 worker 上的 Model Parallelism 及第二代参数服务器架构

- ***(DistBelief) Large Scale Distributed Deep Networks***

    - Abstract: DistBelief 是 Google 在 2012 年提出的一个支持多机分布式训练的软件框架，第一次对大模型提出了模型并行（Model Parallelism）的方法（包括不同 layers 分布在不同 machines 上，以及相同 layer 中的不同子 tensors 分布在不同 machines 上），和数据并行结合，面向在线和批处理场景提出了两类算法。

    - Link: [Note for DistBelief](https://github.com/DicardoX/Notes_for_Papers/tree/main/DistBelief)

------



### 2.6  Automated Hybird Parallelism: Exploration on Other Parallelizing Dimensions

#### 2.6.1 Layer-wise Parallelism Based on Reduction & Search in Computation Graph

- ***(OptCNN) Exploring Hidden Dimensions in Parallelizing Convolutional Neural Networks***

    - Abstract: OptCNN 提出 layer 内部级别的并行，允许每个 layer 有各自的并行策略，通过解决图搜索问题来共优化。OptCNN 能够提高训练吞吐，减少通信开销并达到更好的扩展性。

    - Link: [Note for OptCNN](https://github.com/DicardoX/Notes_for_Papers/tree/main/OptCNN)

#### 2.6.2 SOAP 并行策略搜索空间及基于引导随机和 MCMC 采样实现的增量搜索算法

- ***(FlexFlow) Beyond Data and Model Parallelism for Deep Neural Networks***

    - Abstract: FlexFlow 在 OptCNN 的基础上，提出包括 Sample（layer 的数据并行）、Operator（不同 OP 如何并行）、Attribute（样本高/宽等不同属性如何划分） 和 Parameter（channel 等模型参数如何在设备间分布） 在内的 SOAP 并行策略搜索空间，一个基于 profile 和理论计算的，对特定策略进行性能预测的增量执行模拟器，以及一个面向最优并行策略的基于引导随机和 MCMC 采样的增量搜索算法。

    - Link: [Note for FlexFlow](https://github.com/DicardoX/Notes_for_Papers/tree/main/FlexFlow)

#### 2.6.3 代数变换和并行化在并行计算图中的统一表示及作为图替代的共优化（设备映射独立底层优化）

- ***Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization***

    - Abstract: Unity 在 FlexFlow、TASO 和 MetaFlow 的基础上，提出在并行计算图（PCG）中代数变换和并行化的统一表示（OP，Operator）和共优化（图替代，Substitution）方法，可以同时考虑分布式训练中的计算、并行和通信过程。对于共优化，Unity 使用一个多级搜索算法来高效搜索性能最好的图替代组合以及相应的硬件放置策略。此外，Unity 基于先前工作定义了 DNN 并行系统中常见的六类基本形式。

    - Link: [Note for Unity](https://github.com/DicardoX/Notes_for_Papers/tree/main/Unity)

#### 2.6.4 图替代的生成、验证和剪枝，以及  (MetaFlow) 基于开销的回溯搜索算法

- ***TASO: Optimizing Deep Learning Computation with Automatic Generation of Graph Substitutions***

    - Abstract: TASO 是 Unity 的主要前作，更为详细地阐述了如何进行替代生成、替代验证，以及剪枝冗余的替代，并复用了 MetaFlow 提出的基于开销的回溯搜索算法，并拓展了对 tensor 在内存中 data layout 的枚举优化。

        替代生成：通过 DFS 迭代添加 OP，枚举计算图，再计算两步哈希的指纹，指纹相同的二次验证

        - Step 1. 在现有图中迭代添加 OP，通过枚举 OP 类型及 tensor 输入，通过 DFS 算法来构造所有的非循环且不包含重复计算的计算图（若某个图中存在两个 OP 对相同输入 tensor 进行了相同的计算，则定义为包含重复计算）；同时用随机 tensor 和常数 tensor 作为输入，来查找涉及常数 tensor 的替代（如单位矩阵的 OP，输出等于输入，是一个常数替代）；为了避免计算指纹时的 FLOP 精度损失，所有 tensors 都被表示为整数（已有工作）；两步哈希函数：计算每个计算图的指纹。先对图的每个输出 tensor 考虑 size、shape 和 content 地计算一次哈希，再对所有输出 tensor 的第一次哈希值无序地进行第二次对称哈希；
        - Step 2. 对于拥有相同指纹的两个图，TASO 会进一步在一系列 test case 上检测。每个 case 包含一系列随机输入 tensor，若两个图的输出 tensor 相差小于一个阈值则通过（未和计算指纹时一样使用整数化）；存在两类 OP 需要特殊处理：1) relu，对所有负输入总返回 0，造成许多无用的替代被判定为合法，因此使用一个随机非线性函数来在图计算中替代 relu；2) enlarge，通过 padding，也会造成无用替代被判定为合法，仅考虑将 enlarge 用在输入 tensor 的计算图；

        替代验证：这里的 OP 性质还是人工设置并检测的，Unity 里面也是。

        - OP 性质的验证步骤：TASO 在小范围内所有参数值和 tensor size 上验证 OP 性质，因此需要在 Python 中将每个 tensor OP 表示为基础的符号化实现；然后用 Z3 证明器来验证 OP 性质；

        冗余替代剪枝：图替代是冗余的，若其被一个更加通用的合法替代所包含，其也是剪枝的目标；剪枝操作会保留潜在的优化可能：若图 G 可以通过一系列替代转换为 G'，则剪枝后也可以；

        - Step 1. 消除所有可通过输入 tensor 重命名而等价的替代；
        - Step 2. 相同子图：
            - Type 1. 若合法，将源和目标图内的相同子图替换为一个新的输入 tensor，更加通用；
            - Type 2. 若合法，将源和目标图内包含所有输出的相同子图去除，将该子图的相同输入作为源和目标图的新输出，更加通用；

        基于开销的回溯搜索算法 (MetaFlow) ：使用 MetaFlow 提出的基于开销的回溯搜索算法，并基于源图的数据布局和目标图内 OP 支持的布局，枚举目标图可能的数据布局；cost 模型基于  OP 进行密集线性且无分支代数计算的事实，因此其在硬件上的性能是高度连续且可预测的（给定数据布局和配置参数）；如 MetaFlow 和 FlexFlow 一样，测单个 OP 然后加；图替代可能会导致环的产生，因此在将 G' 入队前要先检查是否有环。

        整篇 paper 最 fancy 的地方在于对图替代生成、验证和冗余替代剪枝的阐述，是 Unity 的基础。

    - Link: [Notes for TASO (& MetaFlow)](https://github.com/DicardoX/Notes_for_Papers/tree/main/TASO_MetaFlow)

