# Notes for Papers

[https://github.com/DicardoX/Notes_for_Papers](https://github.com/DicardoX/Notes_for_Papers)

> This repository is designed to record personal notes for reading papers.

-----



## 1. Deployment of DNN Service

### 1.1 Cluster-level & Co-location

- ***Nexus: A GPU Cluster Engine for Accelerating DNN-Based Video Analysis***
    - Abstract: *(2019 SOSP)*. GPU 集群层面的 DNN 服务调度，考虑混布，在满足低时延的同时实现高利用率
    
    - Link: [Note for Nexus](https://github.com/DicardoX/Notes_for_Papers/tree/main/Nexus)

--------



## 2. Distributed Deep Learning

### 2.1 Data Parallelism: Parameter Server

#### 2.1.1 第三代参数服务器架构

- ***Scaling Distributed Machine Learning with the Parameter Server***
    - Abstract: *(2014 OSDI)*. 第三代参数服务器架构，支持 scale 和 fault tolerance，异步通信，网络参数切分，灵活的一致性模型
    
    - Link: [Note for Parameter Server (3rd)](https://github.com/DicardoX/Notes_for_Papers/tree/main/Parameter_Server_3rd)

#### 2.1.2 面向对分布式训练任务收敛时间及速度的在线预测

- ***Optimus: An Efficient Dynamic Resource Scheduler for Deep Learning Clusters***

    - Abstract: *(2018 EuroSys)*. Optimus 是一个面向分布式训练 job 的调度器，主要目标是最小化 JCT，主要分为对 DL 模型收敛时间的预测，对模型训练速度的评估，以及一个考虑资源分配和 task（相较于 job 的更小级别）放置三个部分。

    - Link: [Note for Optimus](https://github.com/DicardoX/Notes_for_Papers/tree/main/Optimus)

#### 2.1.3 面向无信息和部分信息场景的离散优先级抢占式调度算法及合并放置约束的深入讨论

- ***Tiresias: A GPU Cluster Manager for Distributed Deep Learning***
    - Abstract: *(2019 NSDI)*. Tiresias 是一个面向参数服务器（Parameter Server）架构下分布式训练 job 的调度器，主要包括一个 2DAS 调度器，以及基于模型结构对合并放置约束的放宽策略。
    
    - Link: [Note for Tiresias](https://github.com/DicardoX/Notes_for_Papers/tree/main/Tiresias)

#### 2.1.4 异构集群中探索 intra-job 和 inter-job (非 model slice) 调度方式，同步 PS，支持 GPU 抢占并优化 task switching 开销

- ***Hare: Exploiting Inter-job and Intra-job Parallelism of Distributed Machine Learning on Heterogeneous GPUs***

    - Abstract: *(2022 HPDC)*. 本工作**与 model slice 无关，是针对多 jobs slice 出来的 tasks 在异构集群中的调度方式**。Inter-job parallelism 指不同 jobs 的 tasks 在多个异构 GPUs 上并行；intra-job parallelism 指相同 job 的 tasks 在多个异构 GPUs 上并行。Hare 是一个可以**在异构 GPU 集群中探索 inter-job 和 intra-job 并行方式的 job 调度器**，包括：(1) 利用 DML 调度的特性优化 GPU 执行环境以**减少 task 切换开销** (借鉴 **PipeSwitch** (pipeline model 传输和执行 + 预创建 CUDA context)，**early task cleaning** 在每个 layer 后向完成后即清理，**speculative memory management** 保存 task seq 中已训练完单后面还有相同 job 的 task 的 data)；(2) 一个 **relaxed scal-fixed** (其实就是同步 PS) 的同步策略，允许相同训练轮次内独立 tasks 被灵活调度 (支持抢占)；(3) 一个考虑 job 特性和硬件异构的**快速启发式调度算法**，以**最小化 total 加权 JCT**. (数学建模推导较多，后面可以看一下).

    - Link: [Note for Hare](https://github.com/DicardoX/Notes_for_Papers/tree/main/Hare)

--------



### 2.2 Data Parallelism: Ring All-reduce

#### 2.2.1 通信量优化的 Ring All-reduce 工具

- ***Horovod: fast and easy distributed deep learning in TensorFlow***
    - Abstract: *(2018 arxiv)*. 基于 Baidu Ring Allreduce 框架进行代码实现和改进，python package，使用 Nvidia NCCL 内置的优化版本 ring allreduce，支持单模型在单服务器上的多 GPU 部署，部分 API 改进
    
    - Link: [Note for Horovod](https://github.com/DicardoX/Notes_for_Papers/tree/main/Horovod)

#### 2.2.2 分布式训练时 SGD 的 Large Minibatch 实现，可集成到 Ring All-reduce 方法

- ***Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour***
    - Abstract: *(2017 arxiv)*. Facebook 提出的分布式训练下 Distributed SGD 过程在应用大的 minibatch 的同时，保证训练准确性的方法。
    
    - Link: [Note for Large Minibatch Distributed SGD](https://github.com/DicardoX/Notes_for_Papers/tree/main/Large_Minibatch_Distributed_SGD)

----------



### 2.3 Data Parallelism: Combination of PS and Ring All-reduce

#### 2.3.1 PS 和 Ring All-reduce 算法的统一架构

- ***(BytePS) A Unified Architecture for Accelerating Distributed DNN Training in Heterogeneous GPU/CPU Clusters***
    - Abstract: *(2020 OSDI)*. ByteDance 提出的将 PS 和 Ring All-reduce 两种架构综合考虑的一种统一集群内通信架构，利用集群中空闲的 CPU 和带宽资源，并将 PS 和 Ring All-reduce 成功表述为统一架构下的特殊情况。
    
    - Link: [Note for BytePS](https://github.com/DicardoX/Notes_for_Papers/tree/main/BytePS)

--------



### 2.4 Multi-Model Schedule: Co-optimizing at Job-level and Cluster-level

#### 2.4.1 集群中多任务（模型） Job-level (bs, lr) 和 cluster-level (resource allocation) 的 modeling 和 co-optimization

- ***Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning***
    - Abstract: *(2021 OSDI)*. Petuum 提出的将分布式训练中的任务（模型）粒度的 metric modeling 和集群层面的 resource scheduling 结合起来 co-optimize 的分布式训练和调度架构。注意，与以往工作（如 BytePS）不同的是，Pollux 不再是在考虑集群通信拓扑和算力的前提下被动地适应，而是用 metric modeling 来为 resource allocation 提供依据，进而主动地共优化。
    
    - Link: [Note for Pollux](https://github.com/DicardoX/Notes_for_Papers/tree/main/Pollux)

---------



### 2.5 Model Parallelism: The Foundation of Model Parallelism

#### 2.5.1 局部 worker 上的 Model Parallelism 及第二代参数服务器架构

- ***(DistBelief) Large Scale Distributed Deep Networks***

    - Abstract: *(2012 NeurIPS)*. DistBelief 是 Google 在 2012 年提出的一个支持多机分布式训练的软件框架，第一次对大模型提出了模型并行（Model Parallelism）的方法（包括不同 layers 分布在不同 machines 上，以及相同 layer 中的不同子 tensors 分布在不同 machines 上），和数据并行结合，面向在线和批处理场景提出了两类算法。

    - Link: [Note for DistBelief](https://github.com/DicardoX/Notes_for_Papers/tree/main/DistBelief)

------



### 2.6  Automated Hybrid Parallelism: Exploration on Other Parallelizing Dimensions

#### 2.6.1 Layer-wise Parallelism Based on Reduction & Search in Computation Graph

- ***(OptCNN) Exploring Hidden Dizmensions in Parallelizing Convolutional Neural Networks***

    - Abstract: *(2018 ICML)*. OptCNN 提出 layer 内部级别的并行，允许每个 layer 有各自的并行策略，通过解决图搜索问题来共优化。OptCNN 能够提高训练吞吐，减少通信开销并达到更好的扩展性。

    - Link: [Note for OptCNN](https://github.com/DicardoX/Notes_for_Papers/tree/main/OptCNN)

#### 2.6.2 SOAP 并行策略搜索空间及基于引导随机和 MCMC 采样实现的增量搜索算法

- ***(FlexFlow) Beyond Data and Model Parallelism for Deep Neural Networks***

    - Abstract: *(2019 SysML)*. FlexFlow 在 OptCNN 的基础上，提出包括 Sample（layer 的数据并行）、Operator（不同 OP 如何并行）、Attribute（样本高/宽等不同属性如何划分） 和 Parameter（channel 等模型参数如何在设备间分布） 在内的 SOAP 并行策略搜索空间，一个基于 profile 和理论计算的，对特定策略进行性能预测的增量执行模拟器，以及一个面向最优并行策略的基于引导随机和 MCMC 采样的增量搜索算法。

    - Link: [Note for FlexFlow](https://github.com/DicardoX/Notes_for_Papers/tree/main/FlexFlow)

#### 2.6.3 代数变换和并行化在并行计算图中的统一表示及作为图替代的共优化（设备映射独立底层优化）

- ***Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization***

    - Abstract: *(2022 OSDI)*. Unity 在 FlexFlow、TASO 和 MetaFlow 的基础上，提出在并行计算图（PCG）中代数变换和并行化的统一表示（OP，Operator）和共优化（图替代，Substitution）方法，可以同时考虑分布式训练中的计算、并行和通信过程。对于共优化，Unity 使用一个多级搜索算法来高效搜索性能最好的图替代组合以及相应的硬件放置策略。此外，Unity 基于先前工作定义了 DNN 并行系统中常见的六类基本形式。

    - Link: [Note for Unity](https://github.com/DicardoX/Notes_for_Papers/tree/main/Unity)

#### 2.6.4 图替代的生成、验证和剪枝，以及  (MetaFlow) 基于开销的回溯搜索算法

- ***TASO: Optimizing Deep Learning Computation with Automatic Generation of Graph Substitutions***

    - Abstract: *(2019 SOSP)*. TASO 是 Unity 的主要前作，更为详细地阐述了如何进行替代生成、替代验证，以及剪枝冗余的替代，并复用了 MetaFlow 提出的基于开销的回溯搜索算法，并拓展了对 tensor 在内存中 data layout 的枚举优化。

    - Link: [Note for TASO (& MetaFlow)](https://github.com/DicardoX/Notes_for_Papers/tree/main/TASO_MetaFlow)

#### 2.6.5 面向 OP 粒度的 tensor partion-n-reduce 划分，基于 TDL 的 OP 描述，以及最小化通信开销的 recursive DP 搜索算法

- ***(Tofu) Supporting Very Large Models using AutomaticDataflow Graph Partitioning***

    - Abstract: *(2019 EuroSys)*. Tofu 和 OptCNN 及 FlexFlow 想解决的问题一样，是同时期对自动化并行的探索。相较于 layer-wise，Tofu 以 OP 级别的 tensor 为粒度，将大模型的数据流图以 partition-n-reduce 的方式，等分划分到多个 GPU（仅划分各类 tensor，每个 GPU 都拷贝一份完整的图 OP），以减少 GPU 的内存足迹，同时达到并行化的效果；Tofu 使用一个简单的 Halide-like 语言 TDL 来描述 OP 的语义；在划分 OP 时，Tofu 使用一个 DP 套递归的搜索算法来最小化通信开销。

    - Link: [Note for Tofu](https://github.com/DicardoX/Notes_for_Papers/tree/main/Tofu)

------



### 2.7 Pipeline Parallelism: Considered with Other Dimensions

#### 2.7.1 流水线并行和数据并行混合的 stage 划分、replica 数目以及设备放置 DP 搜索，以及面向流水线内存开销的优化调度

- ***DAPPLE: A Pipelined Data Parallel Approach for Training Large Models***

    - Abstract: *(2021 PPoPP)*. DAPPLE 是一个同步训练框架，将大模型的数据并行（stage-level replica）和流水线并行统一，在保证训练收敛性的同时提高内存效率。DAPPLE 由 DAPPLE profiler**，**DAPPLE planner 和 DAPPLE runtime 组成。Planner 尝试求解 stage 划分，replica 数目和设备放置问题，探索数据和流水线并行的最优混合策略；runtime 包括一个基于依赖关系的 early backward scheduling & warmup 调度算法，在减少设备内存使用的同时保证不影响吞吐。

    - Link: [Note for DAPPLE](https://github.com/DicardoX/Notes_for_Papers/tree/main/DAPPLE)

#### 2.7.2 流水线并行和 OP (data, model, ZeRO, etc) 并行混合的 intra-op (ILP 优化) + inter-op (DP 优化) 两级并行编译架构

- ***Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning***

    - Abstract: *(2022 OSDI)*. Alpa 将 pipeline parallelism 和已有并行方法相统一，提出 intra-op (data, model, ZeRO, etc) 和 inter-op (pipeline) 的两级并行架构。在 intra-op level，Alpa 通过整数线性规划 (ILP, merge simple op) 优化 intra-op parallelism，通过通信量/带宽来评估通信 & reshard 开销，最小化 stage（包含多 OPs）在给定 device mesh 上的执行开销；在 inter-op level，Alpa 基于 intra-op pass 返回的 cost，基于 intra-op plan 将 stage-mesh pair 编译为可执行文件，profile 以获得精确的 stage latency，stage 所需内存和中间激励存储的所需内存，通过一个动态规划 (DP) 算法 (early-pruning & op clustering) 考虑如何切分模型/设备集群为 stages/device meshes，和如何映射为 stage-mesh pair，最小化 end-to-end pipeline latency。

    - Link: [Note for Alpa](https://github.com/DicardoX/Notes_for_Papers/tree/main/Alpa)

#### 2.7.3 基于 user 的少量 tensor sharding 标注，intra-op 混合并行，将 pipeline 规约为 tensor 划分问题

- ***GSPMD: General and Scalable Parallelization for ML Computation Graphs***

    - Abstract: GSPMD 是一个自动化，compiler-based 的并行系统，允许 users 通过少量指示如何 distribute tensors 的标注，像单机那样编程。GSPMD 的 OP 划分表示简单有效，可表示不同或混合类别的并行，包括 DP，in-layer MP，spatial parallelism 和 weight update sharding，并通过一个 wrapper library 将 pipeline reduce 为一个 tensor/OP 划分问题。

    - Link: [Note for GSPMD](https://github.com/DicardoX/Notes_for_Papers/tree/main/GSPMD)

---------



### 2.8 Weight Update Sharding: Partition Optimizer States (Momentum, Variance, FP32 Weights), Gradients and Parameters

#### 2.8.1 面向内存显著优化和有限额外通信开销的 DP worker 优化器状态、梯度和参数共享方法

- ***ZeRO: Memory Optimization Towards Training A Trillion Parameter Models***

    - Abstract: *(2020 SC)*. ZeRO 优化器提出了 weight update sharding 方法，可以优化内存，消除 data parallelism 和 model parallelism 的内存冗余；同时，ZeRO 可以在加快训练速度的同时，增大可有效训练的模型 size，并使其与设备数目成比例增加；此外，ZeRO 可以保持较低的额外通信开销（为了换取内存优化），且指出 ZeRO 对于通信延迟的影响较小，且相较于通信量和通信带宽，延迟对训练速度的限制更小。

    - Link: [Note for ZeRO](https://github.com/DicardoX/Notes_for_Papers/tree/main/ZeRO)

---------



### 2.9 Resource Allocation Framework

#### 2.9.1 基于 DLT job 资源使用周期可预测性，profile 和贪心启发式的资源动态分配

- ***Gandiva: Introspective Cluster Scheduling for Deep Learning***

    - Abstract: *(2018 OSDI)*. Gandiva 是一个集群调度框架，通过利用 profile 信息来优化延迟，提高训练效率和集群利用率。考虑如下 DL 特点：1) Feedback-driven exploration -> simultaneous early feedback；2) DL jobs 在资源利用上的异构性 (locality & interference)；3) Intra-job predictability。Gandiva 基于以上特点，设计 suspend-resume (time-slice with weighted round robin), pack (对应单 GPU 混布多任务), migration (去碎片化), grow-shrink (提高集群利用率 & 为新 job 提供资源以共优化) 等机制，通过 profile 监控：1) 资源使用情况；2) job progress rate (mini-batch time) 来判断调度策略的有效性。在调度策略上 (reactive mode -> 事件处理，introspective mode -> 持续监听并优化 job replacement)，Gandiva 倾向于将 job 分配到相同 affinity (相同需占 GPU 数目) 的 server 上，并通过 over-subscription (time-slice & suspend-resume) 来尽可能消除新 job 的排队时间，保证 early feedback；基于 profile 和一个贪心启发式算法，利用 packing, migration, grow-shrink 等方法来 runtime 优化 cluster efficiency (包括 util)。注意，Cluster fairness (servers 间 load 均衡) 并不考虑，仅考虑 server 内 job fairness (time-slice weighted RR 来避免 job 长时间等待)。

    - Link: [Note for Gandiva](https://github.com/DicardoX/Notes_for_Papers/tree/main/Gandiva)

#### 2.9.2 Co-design Cluster Scheduler 和 DL 框架，动态 scale 资源量，单 GPU 混布多 jobs

- ***AntMan: Dynamic Scaling on GPU Clusters for Deep Learning***
    - Abstract: *(2020 OSDI)*. AntMan 是一个 **co-design cluster scheduler 和 DL 框架的深度学习架构**，利用 DL training 的特性，在 **local coordinator** 中引入 **memory 和 computation 动态 scale 机制**，收集 **DL 框架和硬件的相关信息**，利用 **GPU Operator Manager 动态管理 Resource-Guarantee (RG) jobs 和 Opportunistic (OT) jobs 执行流的 GPU sharing**，避免对 RG jobs 的性能干扰；在 **global scheduler** 中为每个用户**维护一个支持 job arrival 的队列**，并分别**以不同的策略调度 RG jobs 和 OT jobs**，从而**分配 GPU 资源** (GPU 内存，计算单元)。

    - Link: [Note for AntMan](https://github.com/DicardoX/Notes_for_Papers/tree/main/AntMan)
