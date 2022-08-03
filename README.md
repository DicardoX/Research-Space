# Notes for Papers

[https://github.com/DicardoX/Notes_for_Papers](https://github.com/DicardoX/Notes_for_Papers)

> This repository is designed to record personal notes for reading papers.

-----



## 1. Deployment of DNN Service

### 1.1 Cluster-level & Co-location

- ***Nexus: A GPU Cluster Engine for Accelerating DNN-Based Video Analysis***
    - Abstract: GPU 集群层面的 DNN 服务调度，考虑混布，在满足低时延的同时实现高利用率
    - Link: [Notes for Nexus](https://github.com/DicardoX/Notes_for_Papers/tree/main/Nexus)

--------



## 2. Distributed Deep Learning

### 2.1 Data Parallelism: Parameter Server

#### 2.1.1 第三代参数服务器架构

- ***Scaling Distributed Machine Learning with the Parameter Server***
    - Abstract: 第三代参数服务器架构，支持 scale 和 fault tolerance，异步通信，网络参数切分，灵活的一致性模型
    - Link: [Notes for Parameter Server (3rd)](https://github.com/DicardoX/Notes_for_Papers/tree/main/Parameter_Server_3rd)

#### 2.1.2 面向对分布式训练任务收敛时间及速度的在线预测

- ***Optimus: An Efficient Dynamic Resource Scheduler for Deep Learning Clusters***

    - Abstract: Optimus 是一个面向分布式训练 job 的调度器，主要目标是最小化 JCT，主要分为对 DL 模型收敛时间的预测，对模型训练速度的评估，以及一个考虑资源分配和 task（相较于 job 的更小级别）放置三个部分。

        - Optimus 对 DL 模型收敛时间的预测基于 SGD 收敛速率为 O (1 / k) 的事实（该结论对学习率的变化有要求），使用 non-negative least squares (NNLS) 求解器来进行在线的拟合。
        - 为了建立性能评估模型，Optimus 建立了数学化分布式训练任务和资源的系统模型，并分析了包括 worker 前向（与 bs 相关） & 后向（与 bs 无关）时间，数据传输时间（部分模型大小 / 带宽），parameter server (ps) 的参数更新时间，以及通信开销（与 worker 和 ps 的数目成线性关系）在内的总时间 T。基于上述分析，Optimus 构造了一个表示训练速度 f = 1 / T 的函数，注意该函数以 worker 和 ps 的数目为自变量，对于同步和异步训练有些许区别（前者需要进一步考虑用户对 overall batch size M 的输入），并以 NNLS 拟合的方式，通过预训练和在线收集训练数据建立和不断改进模型。
        - 最后，基于任务剩余的代数以及上面得到的训练速度函数 f，Optimus 定义了边际收益，来以启发式（贪心）的方式进行资源分配（不用整数规划是因为非线性甚至非凸，且 NP-hard）。
        - 对于 task 放置（实际上是对 job 所被分配的 workers 和 ps 进行 servers 资源的分配，分配完后均衡地放在这些 servers 上），Optimus 分析了达到最小化最大数据传输的方式，包括使用最少的 servers 来部署以及均衡地在这些 servers 上面放置 workers 和 ps，并以此提出了基于贪心的分布式策略。此外，为了解决 stragglers 的问题，Optimus 分别对 workers 的训练速度进行均衡，并提出了对 ps 负载进行均衡的方法。
    
        然而，Optimus 对模型训练时间的预测在其他工作中被证明为过分简化了损失函数的变化曲线，在实际集群中并不总是适用的。
    
    - Link: [Notes for Optimus](https://github.com/DicardoX/Notes_for_Papers/tree/main/Optimus)

#### 2.1.3 面向无信息和部分信息场景的离散优先级抢占式调度算法及合并放置约束的深入讨论

- ***Tiresias: A GPU Cluster Manager for Distributed Deep Learning***

    - Abstract: Tiresias 是一个面向参数服务器（Parameter Server）架构下分布式训练 job 的调度器，主要包括一个 2DAS 调度器，以及基于模型结构对合并放置约束的放宽策略。首先，Tiresias 讨论了同时考虑时间（JCT）和空间（资源分配）两个维度（2D）以计算优先级的重要性，以及优先级离散化的优势。Tiresias 进行架构设计的前提条件包括：DL job 的执行时间通常是不可预测的，job 的特点无法提前获知，以及同步 PS 架构下 DL job 的资源分配具有 all-or-nothing 的特点。

        - 对于 2DAS 调度器：当无先验信息时，优先级函数采用 2D-LAS 算法，job 获取的资源量等于当前执行时间 * 分配资源量；若提供部分先验信息（比如执行几轮后，但 paper 中并没有说明如何通过几轮的信息拟合得到 JCT 分布，只说需要管理者提供该分布），优先级的值则与 Gittins index value 相等，Gittins index 值代表该 job 获取一定量服务后在拿到下一个服务量后能完成的可能性，值越高表示优先级越高。
        - 对于优先级离散化：参考多级反馈队列（MLFQ）的架构进行设计。LAS时，对相同队列里的任务采用基于 start time 的 FIFO；Gittins index 中，service quantum 被设置为当前队列的服务量上界，当 job 消耗完 \Delta 时被降级。相同队列中的 jobs 以各自的 Gittins index 值来调度，最低队列（\Delta 为无穷，无法计算 Gittins index value）以 FIFO 的方式。
        - 对于合并放置约束：Tiresias 使用模型结构的倾斜程度来预测 jobs 是否对合并放置敏感。部分模型的某些层是很大的 tensors，而在模型聚合时该层参数的信息大小与 tensors 大小相关。因此，聚合大 tensors 更容易受到网络竞争的影响。由于每个 parameter server 都会周期性地向每个 worker 发出自己拥有的更新后的模型参数，Tiresias 通过监控网络通信来获取模型倾斜信息，并构造了一个自己的监控工具。

        整篇 paper 最 fancy 的地方在于面向无 JCT 分布信息和部分信息抢占式 2D 调度算法设计，参考多级反馈队列设计的优先级离散化架构，以及基于网络通信对模型 tensors 倾斜信息的监控方法。

--------



### 2.2 Data Parallelism: Ring All-reduce

#### 2.2.1 Ring All-reduce 工具

- ***Horovod: fast and easy distributed deep learning in TensorFlow***
    - Abstract: 基于 Baidu Ring Allreduce 框架进行代码实现和改进，python package，使用 Nvidia NCCL 内置的优化版本 ring allreduce，支持单服务器上的多 GPU 部署，部分 API 改进
    - Link: [Notes for Horovod](https://github.com/DicardoX/Notes_for_Papers/tree/main/Horovod)

#### 2.2.2 分布式训练时 SGD 的 Large Minibatch 实现，可集成到 Ring All-reduce 方法

- ***Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour***
    - Abstract: Facebook 提出的分布式训练下 Distributed SGD 过程在应用大的 minibatch 的同时，保证训练准确性的方法，建立学习率与 batch size 的线性函数关系，每个 GPU/worker 的 bs 确定，通过调整 GPU 数目来改变整体的 minibatch size。由于将求梯度时的 average 操作限制在 per-worker 级别，可以集成到 ring allreduce 等仅支持加法操作的梯度聚合方法上。在通信方面，本文针对梯度聚合提出了三步 allreduce 操作，依次是相同 server 上 GPUs 的 buffers 先加（为了接近线性 scale，由于不同 layers 之间的梯度没有依赖关系，可以将相同 server 上 workers 的梯度聚合和 bp 并行开展），不同 servers 上的 result buffers 再加，结果广播给每个 GPU。servers 间的通信主要包括两个算法：
    
        - 算法一：recursive halving and doubling 算法。包括 log 级别的通信步数，相对更快。可以视为经典的二叉树形算法。
        - 算法二：bucket (ring) 算法。包括一次方级别的通信步数，相对更慢。
    
        一点启发是，可以通过 profile 小的 minibatch，来评估大的 minibatch。
    
    - Link: [Notes for Large Minibatch Distributed SGD](https://github.com/DicardoX/Notes_for_Papers/tree/main/Large_Minibatch_Distributed_SGD)

----------



### 2.3 Data Parallelism: Combination of PS and Ring All-reduce

#### 2.3.1 PS 和 Ring All-reduce 算法的统一架构

- ***(BytePS) A Unified Architecture for Accelerating Distributed DNN Training in Heterogeneous GPU/CPU Clusters***
    - Abstract: ByteDance 提出的将 PS 和 Ring All-reduce 两种架构综合考虑的一种统一集群内通信架构，利用集群中空闲的 CPU 和带宽资源，并将 PS 和 Ring All-reduce 成功表述为统一架构下的特殊情况。SS（Summation Service）只运行在 CPU 中（包括 CPU 机器和 GPU 机器），负责从 CS 侧接收 tensors，把 tensors 加起来，再返回给 CS；CS (Communication Service) 只运行在 GPU 中，负责同步多个局部 GPU 之间的 tensors。该架构能够根据集群中可用 CPU 和 GPU 的相对数目，通过硬件 profile 动态决定 SS 中分配在 CPU / GPU 的数据的比例，这样得到的机器间通信是延迟最优的。同时，BytePS 研究了机器内拓扑结构对通信效率的影响，针对 PCIe-only 拓扑提出了 CPU-assisted aggregation 策略（可 pipeline），针对 NVLink-based 拓扑使用 reduce and broadcast 策略。BytePS 使 CPU 仅负责 sum（SS），而 GPU 负责 FP、BP 和 parameter update。由于在 sum 之前 GPU 就要 update 参数，因此破坏了 PS 原有对异步并行的支持性， BytePS 因此提出支持异步的参数更新算法，即向 CPU 传输 delta w，并证明了其和 PS 异步并行的等效性。
    - Link: [Notes for BytePS](https://github.com/DicardoX/Notes_for_Papers/tree/main/BytePS)

--------



### 2.4 Data Parallelism: Co-optimizing at Job-level and Cluster-level

#### 2.4.1 Job-level (bs, lr) 和 cluster-level (resource allocation) 的 modeling 和 co-optimization

- ***Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning***

    - Abstract: Petuum 提出的将分布式训练中的 metric modeling 和 resource scheduling 结合起来 co-optimize 的分布式训练和调度架构。Pollux 可以同时用在 parameter server 和 all-reduce 两类分布式架构，原因是重点关注 goodput 为主的性能指标，而非 ps 和 all-reduce 等架构中最重要的通信量的优化，这也是 Pollux 可以优化（考虑不足）的一个点。考虑的主要 metrics 为 statistical efficiency（表征每一轮迭代模型精度能够进步多少） 和 throughput，并分别对这两个 metric 的 estimate 进行了 modeling。Pollux 使用 gradient accumulation 来进一步扩大 batch size，经过 s 轮的 GPU 本地梯度聚合后再进行全局的梯度聚合。基于这两个 metric，Pollux 提出了自己的调度和优化 metric，即每个job 的 goodput。整个 Pollux 的架构分为两层：

        -  job 层面，PolluxAgent 收集 bs 和 $T_{iter}$ 等信息，基于信息拟合 efficiency 和 throughput 的函数，进而获取每个 job 的 goodput 函数；通过最大化 job 的 goodput 来动态调整 bs 和 lr，以更好地利用资源。最后，周期地向 PolluxSched 报告 goodput 函数，等待新一轮资源分配后再调整 bs 和 lr。
        - Cluster 层面：PolluxSched 基于 jobs 的 goodput 动态重分配资源，通过最大化 fitness 函数来获取理论最优的分配，并考虑多个集群层面的目标，包括 fairness，goodput，reallocataion overhead，inference slowdown 等

        整篇 paper 最 fancy 的地方在于它的建模过程，包括 metric modeling 和 scheduling (optimization) modeling。

    - Link: [Notes for Pollux](https://github.com/DicardoX/Notes_for_Papers/tree/main/Pollux)

---------



### 2.5 Model Parallelism: The Foundation of Model Parallelism

#### 2.5.1 局部 worker 上的 Model Parallelism 及第二代参数服务器架构

- ***(DistBelief) Large Scale Distributed Deep Networks***

    - Abstract: DistBelief 是 Google 在 2012 年提出的一个支持多机分布式训练的软件框架，第一次对大模型提出了模型并行（Model Parallelism）的方法，包括不同 layers 分布在不同 machines 上，以及相同 layer 中的不同子 tensors 分布在不同 machines 上，只有包含跨越 machines 的边的 nodes 才需要在 machines 间传输状态。注意，DistBelief 框架也能够支持数据并行，且揭示了异步 SGD（之前很少用在非凸问题）在分布式训练中表现良好，特别是和 Adagrad 自适应学习率方法结合时。

        DistBelief 主要由两个算法构成。算法一是 Downpour SGD，一个异步 SGD 过程，能自适应学习率，支持大规模模型副本；算法二是 Sandblaster L-BFGS，L-BFGS 的分布式实现，同时使用数据和模型并行。

        - Downpour SGD 算法：在线场景。SGD的传统公式本质上是顺序的，因此不适用非常大的数据集。Downpour SGD 就是异步 parameter server (PS) 架构的模型并行版本。不同的 workers（都保存一份独立的 replica，用独立的数据集进行 fp 和 bp）之间和不同的 parameter server shards 之间都是异步的。对于 workers 被划分到不同 machines 上，因此每个 machine 只需跟一部分 ps 通信。注意，machine 发送和接收时需要进行同步，以 replica 为单位，尽量避免（但没有保证）算力差别导致的异步累积，即 ps shards 之间的参数更新代数不同，这会带来更多的随机性。放宽一致性在非凸问题中并无理论依据，但在实际中非常有效。对于 ps shards，异步体现在更新更快的 shards 先把这部分更新后的参数返回给 worker 中特定的 machine。Adagrad 自适应学习率策略直接在 ps shard 中，用梯度来算学习率，易实现。该策略能够扩展 model replicas 能实际使用的个数，且与 “少数几个 replicas warmstarting，再逐步加入其他 replicas” 策略结合使用时，能够很好地解决 Downpour SGD 训练时的稳定性问题。

        - Sandblaster L-BFGS 算法：批处理场景。Sandblaster 框架下，优化算法（如 L-BFGS）在协调进程中，该进程并不直接访问模型参数，而是向 ps shards 发送一系列子操作（点积，放缩，考虑系数的加，乘等），依次进行一个个 batch 处理的优化，计算结果保存在 shard 本地。这样做可以避免需要把所有参数和梯度都汇聚到一个中心 server 上，这也是模型并行和数据并行 replica machines 和 ps shards  “多对多” 的优势。

            为了缓解短板效应，提出了一个负载均衡策略：协调器给每个 replica 分配一个很小比例的工作（相较于 1/N batch），并给那些完成比较快的 replicas 分配新的更多工作。对于 batch 最后的工作，协调器会让多个 replicas 同时运算，并采用完成最快的那个。

            上述策略意味着，Sandblaster L-BFGS 无需像 Downpour SGD 那样将 dataset 划分为多个独立的 subset，而是以一个个 batch 的形式，在协调器的指挥下，供多个 replicas 进行处理。

        整篇 paper 最 fancy 的地方在于第一个提出了模型并行的设计，并与数据并行相结合（优势在于避免把所有参数和梯度都汇聚到一个中心 server 上），且分别对在线和批处理两类情景提出了两类算法。

    - Link: [Notes for DistBelief](https://github.com/DicardoX/Notes_for_Papers/tree/main/DistBelief)

------



### 2.6  Automated Hybird Parallelism: Exploration on Other Parallelizing Dimensions

#### 2.6.1 Layer-wise Parallelism Based on Reduction & Search in Computation Graph

- ***(OptCNN) Exploring Hidden Dimensions in Parallelizing Convolutional Neural Networks***

    - Abstract: OptCNN 提出 layer 级别的并行，允许每个 layer 有各自的并行策略，通过解决图搜索问题来共优化。OptCNN 能够提高训练吞吐，减少通信开销并达到更好的扩展性。对于模型并行，OptCNN 仅针对每个 layer 的参数并行，而不研究 layer (op) 级别的并行。在处理 2D 图像的标准 CNN 中，数据一般为四维张量 (sample, height, width, channel)。Sample 包括训练集中的 idx（形状意义上为 bs），height 和 width 是图像中的位置（形状意义上为高和宽），channel 是该位置神经元的索引（形状意义上为通道数）。探索这些其他维度的并行策略包括如下优势：

        - 优势一：不同的 layer 采用不同维度的并行策略，可能会有更好的性能表现；
        - 优势二：探索其他维度的并行策略能够减少通信开销；
        - 优势三：不同 layers 对并行度（并行机器数目）的偏好不同，进而影响性能

        一个其他维度并行策略的例子是无需参数同步，只有数据传输的 channel 并行。这里数据传输里 input tensors 最大不会超过输入 batch （比如小于32MB）。全连接层约等于（包括）卷积层，因此这里 channel 并行是每个卷积核的将一半的 channels （共 N 个 channels）分别存储在两个 GPU 上，计算这层卷积时某个 sample 一半 channel 和 GPU 1 上的卷积核 k channel 卷积（得到 N /2 个值），另一半 channel 和 GPU 2 上的卷积（得到 N / 2 个值），然后再通过数据传输加起来，这就是全连接层其中的一个值。若一共 K 个卷积核，则全连接层的输出维度为 K。

        OptCNN 通过设备图（对可用设备及连接建模）和计算图（定义映射到设备图上的 NN）来定义并行问题，通过定义输出张量如何被划分来描述 layer 的并行化，通过并行配置定义该 layer 如何在多设备间并行。OptCNN 为 layer 定义了三类时间：

        - t_c 同时包括 fp 和 bp 的时间，通过当前配置下 layer 在设备上处理多次并取平均 t 得到；
        - t_x 表示将输入张量传输到目标设备的时间，使用数据大小和已知的通信带宽；
        - t_s 表示 bp 后该 layer 同步参数的时间，适用于面向 layer 的 PS 架构（与数据并行的 ps 架构区别），用通信时间近似参数同步时间（忽略 ps 上的更新时间）

        OptCNN 的优化目标是最小化 t_o = sum_{i}(t_c + t_x + t_s)。OptCNN 提出了两类 reduction 方法：

        - 方法一：Node elimination。将计算图中入度和出度均为 1 的 layer 去掉，用 layer 上的 t_c，t_s 以及进出两条边的 t_x 来定义新产生 edge 的 t_x，并使用动态规划来为该 layer 寻找一个最优的配置 c，以使得新的 t_x 最小。如果去掉 layer 后的策略 S 最优，则加上 layer 的配置 c_j 后的 S' 也最优；
        - 方法二：Edge elimination。将计算图中源和目标节点都相同的两条 edge 合并为一条，新 t_x 等于原来两个 t_x 的和。新的策略 S' 和老的策略 S 相同，若 S 是 G 的最优策略，则 S' 是 G' 的最优策略

        OptCNN 采用基于动态规划的图搜索算法。算法流程是：首先迭代调用 node 和 edge elimination，直到计算图无法 reduced；然后枚举获得计算图G(m) 的最优策略 S(m)，以最小化 cost 函数（即总时间）t_o；随后迭代反向撤销每次 elimination（若是 node eli，则最小化 t_o 获得 c，且 S(i) = S(i+1) + c，否则为 edge eli，S(i) = S(i+1)）由定理知 S(i) 是 G(i) 的最优策略；最终 S(0) 即原计算图的最优策略。

        整篇 paper 最 fancy 的地方在于定义了 CNN 多维度并行策略的方法，以及基于多阶段时间和计算图的有效图搜索算法。

    - Link: [Notes for OptCNN](https://github.com/DicardoX/Notes_for_Papers/tree/main/OptCNN)

