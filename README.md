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

        OptCNN 通过设备图（对可用设备及连接建模）和计算图（定义映射到设备图上的 NN）来定义并行问题，通过定义输出张量如何被划分来描述 layer 的并行化，通过并行配置定义该 layer 如何在多设备间并行。OptCNN 为 layer 定义了三类时间，进而通过 simulation 评估特定策略的时间性能：

        - t_c 同时包括 fp 和 bp 的时间，通过当前配置下 layer 在设备上处理多次并取平均 t 得到；
        - t_x 表示将输入张量传输到目标设备的时间，使用数据大小和已知的通信带宽；
        - t_s 表示 bp 后该 layer 同步参数的时间，适用于面向 layer 的 PS 架构（与数据并行的 ps 架构区别），用通信时间近似参数同步时间（忽略 ps 上的更新时间）

        OptCNN 的优化目标是最小化 t_o = sum_{i}(t_c + t_x + t_s)。OptCNN 提出了两类 reduction 方法：

        - 方法一：Node elimination。将计算图中入度和出度均为 1 的 layer 去掉，用 layer 上的 t_c，t_s 以及进出两条边的 t_x 来定义新产生 edge 的 t_x，并使用动态规划来为该 layer 寻找一个最优的配置 c，以使得新的 t_x 最小。如果去掉 layer 后的策略 S 最优，则加上 layer 的配置 c_j 后的 S' 也最优；
        - 方法二：Edge elimination。将计算图中源和目标节点都相同的两条 edge 合并为一条，新 t_x 等于原来两个 t_x 的和。新的策略 S' 和老的策略 S 相同，若 S 是 G 的最优策略，则 S' 是 G' 的最优策略

        OptCNN 采用基于动态规划的图搜索算法。算法流程是：首先迭代调用 node 和 edge elimination，直到计算图无法 reduced；然后枚举获得计算图G(m) 的最优策略 S(m)，以最小化 cost 函数（即总时间）t_o；随后迭代反向撤销每次 elimination（若是 node eli，则最小化 t_o 获得 c，且 S(i) = S(i+1) + c，否则为 edge eli，S(i) = S(i+1)）由定理知 S(i) 是 G(i) 的最优策略；最终 S(0) 即原计算图的最优策略。

        整篇 paper 最 fancy 的地方在于定义了 CNN 多维度并行策略的方法，基于多阶段时间的策略性能模拟评估，以及计算图的有效图搜索算法。

    - Link: [Notes for OptCNN](https://github.com/DicardoX/Notes_for_Papers/tree/main/OptCNN)

#### 2.6.2 SOAP 并行策略搜索空间及基于引导随机和 MCMC 采样实现的增量搜索算法

- ***(FlexFlow) Beyond Data and Model Parallelism for Deep Neural Networks***

    - Abstract: FlexFlow 在 OptCNN 的基础上，提出包括 Sample（layer 的数据并行）、Operator（不同 OP 如何并行）、Attribute（样本高/宽等不同属性如何划分） 和 Parameter（channel 等模型参数如何在设备间分布） 在内的 SOAP 并行策略搜索空间，一个基于 profile 和理论计算的，对特定策略进行性能预测的增量执行模拟器，以及一个面向最优并行策略的基于引导随机和 MCMC 采样的增量搜索算法。

        关于并行维度、配置、策略的定义与 OptCNN 类似，但从面向 CNN 拓展到面向全体 DNN，考虑 OP 维度并行体现在，并行配置中考虑了不同 task 位于哪个设备。FlexFlow 在 OptCNN 之外还假设设备以 FIFO 处理任务（为 OP 维度并行准备）。

        与 OptCNN 不同的是，FlexFlow 在设计图搜索策略时没有局限在 OP 级别，而是以 task 为粒度构造了更为细致的任务图（Task Graph）。任务图对从 OP 中产生的独立任务间的依赖进行建模。将硬件连接建模为通信设备，仅执行通信任务（如数据传输）；计算设备执行普通的计算任务。由于所有设备独立，可进行任务重叠。与 OptCNN 类似，对特定配置下 OP （task）的执行时间（t_c、t_x、t_s）进行建模（profile & 理论计算），并将预测信息整合到任务图中，作为相应任务的执行时间信息。任务图中，节点是任务（计算或通信任务），边是依赖关系。注意，边仅是顺序约束，并不代表数据流（数据流作为通信任务被包括在任务图中）。任务图的构造有如下规则：

        - 规则一：对于配置 c 的 OP，划分为 |c| 个任务作为任务图内的节点；
        - 规则二：对于每个 tensor (op_i, op_j)，分别计算 op_i 和 op_j 各自子任务的 sub-tensors。若 t_i 和 t_j 有共享 tensor 且两个任务在相同设备上，则在任务图中添加边 (t_i, t_j)；若位于不同设备，则先添加通信任务 t_c，再添加 (t_i, t_c) 和 (t_c, t_j) 两条边。t_c 被分配在 t_i 和 t_j 所在设备中间的哪个通信设备上

        相较于使用 Full Simulation 进行系统性能的模拟评估， FlexFlow 使用 Delta Simulation 增量模拟算法。具体来说，FlexFlow 从原来的任务图开始，仅重新模拟执行时间流改变的那部分任务。

        基于上述 SOAP 任务图的构建以及增量模拟方法，FlexFlow 针对不同的策略 S，给出预测的时间性能。

        对于最优并行策略的搜索算法，FlexFlow 采用基于引导随机和 MCMC 采用的增量搜索。MCMC 采样维护现有策略 S，随机提出一个新策略 S'，并根据 cost 模型（即 simulator 模拟得到的预测性能）来计算被采用的可能性。MCMC 表现为贪心搜索算法，倾向于更低的 cost，但也可以跳出局部最优解。FlexFlow 的搜索算法随机选择原有策略 S 的一个 OP，随机重新生成一个配置。利用上述策略提出（proposal）的方法，FlexFlow 将已有策略（如专家定制的策略，数据并行策略等）和一个随机生成的策略同时作为搜索的初始策略，开始搜索（迭代调用 proposal 和 MCMC 采样）直到搜索停止。

        整篇 paper 最 fancy 的地方在于定义了面向全体 DNN 的 SOAP 多维并行策略搜索空间，以 task 为粒度的任务图的构建，以及基于引导随机和 MCMC 采样的增量搜索算法。

    - Link: [Notes for FlexFlow](https://github.com/DicardoX/Notes_for_Papers/tree/main/FlexFlow)

#### 2.6.3 代数变换和并行化在并行计算图中的统一表示及作为图替代的共优化（设备映射独立底层优化）

- ***Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization***

    - Abstract: Unity 在 FlexFlow、TASO 和 MetaFlow 的基础上，提出在并行计算图（PCG）中代数变换和并行化的统一表示（OP，Operator）和共优化（图替代，Substitution）方法，可以同时考虑分布式训练中的计算、并行和通信过程。对于共优化，Unity 使用一个多级搜索算法来高效搜索性能最好的图替代组合以及相应的硬件放置策略。

        Unity 基于先前工作，定义了 DNN 并行系统中常见的六类基本形式：

        - 数据并行（Data Parallelism）：最基本的并行形式；
        - 模型并行（Model Parallelism）：将 DNN 模型拆分为多个子模型，分别在特定的设备训练；
        - 空间并行（Spatial Parallelism）：对 tensor 的空间维度（如图像的高/宽）进行划分；
        - 规约并行（Reduction Parallelism）：利用 OP 的线性。将 tensor 划分为 n 份到 n 个设备上，分别进行 OP 操作。注意每个结果 tensor 的 shape 都和最终 tensor 相同，因此只需相加进行规约；
        - 管道并行（Pipeline Parallelism）：不同训练代数之间的并行；
        - 指定 OP 的并行（Operator-specific Parallelsim）：对 batch 内不同的输入 sample 有不同的 weight，不需要 tensor 备份或同步；

        许多并行策略是不同 cost 指标的 trade-off，因此需要不同 OP 用不同的并行方式，以达到最优性能。DNN 结构表示的传统方法是计算图，节点是 OP，边是 tensor，代数变换被表示为迭代的图替代，通过给每个节点分配并行注释来并行化模型。然而，该方法存在如下局限：

        - 对代数变换和并行化使用独立表示阻碍了共优化。代数变换会增删节点，但并行化需要静态计算图；
        - 计算图并没有显式考虑并行带来的通信开销。这导致代数变换难以预测最终模型的性能

        PCG 的构造规则：在 PCG 中，节点表示并行 OP（表示已有的并行策略，显式表示数据移动和相应的开销）和 tensor OP，边表示 tensor 的数据移动和数据依赖关系。每个 OP 均绑定一个设备映射。

        Tensor 的表示方法：按数据维度对 tensor 进行建模，包含 size 和 degree（该维度划分数）字段。每个 tensor 还有一个 replica 维度，表示该 tensor 数据的副本数。

        设备映射：n 维数组来表示每个 OP 每份运算在哪个设备上运行，n 是并行维度。例如，若 n=3，则将设备组织为三维阵列并序列化，以供该 OP 划分后任务到设备的映射。维度也和设备的拓扑结构相关，一个例子是，模型并行优先在连接相同 CPU 的 GPU 上进行，规约并行优先在相同节点但不同 CPU 上的 GPU 进行，数据并行优先在不同节点间进行。Unity 预设了多种有效利用设备并行的设备映射，开发者也可为特定硬件结构注册自定义映射。例如为节点对在映射种添加额外维度。

        并行 OP：Unity 主要考虑六类并行策略，两两分为三大类：

        - 划分（Partition）和组合（Combine）：可以改变 tensor 除 replica 以外维度的并行 degree；
        - 复制（Replication）和规约（Reduce）：可以改变 tensor replica 维度的并行 degree；
        - 管道（Pipeline）和批处理（Batch）：管道将 tensor 沿某个维度等大小地划分为多个子 batch，串行处理（并未更改 tensor 的并行 degree，weight 需要和子 batch 相同 size）；批处理将多代的 tensors 聚合在一起

        图替代：共优化的有效性依赖于选择一组合适的图替代。Unity 将大而复杂地代数变换和并行策略表示为多个小 PCG 替代的组合。Unity 使用 TASO 提出的超优化方法，先用启发式来生成并识别候选替代，再用开销更大的验证方法来检验替代的正确性。

        - 替代生成：Unity 在固定图 size 内枚举所有可能的 PCG（忽略过大的，不影响能应用变换的 size，可组合）。对每个生成的 PCG，Unity 计算一个指纹（Fingerprint，输入特定 tensor 时输出 tensor 的哈希），该指纹函数被拓展以包含每个维度的并行 degree。若一对 PCG 有相同的指纹，则是一个候选替代；
        - 替代验证：可离线。Unity 使用自动理论证明器 Z3。其中，OP 性质以一阶逻辑演算的方式被提供，OP 表述为输入和并行度的函数（Partition 等可能还需要提供并行维度的信息）。尝试用 Z3 验证所有候选替代，若某个替代正确但不能被验证，则添加缺失的 OP properties，重复直到 Unity 引入的全部新替代都被验证。

        共优化：核心问题是给定 1) PCG；2) OP 级别设备映射的集合；3) PCG 替代的集合，找 1) PCG 替代的序列；2) 结果 PCG 的设备映射，从而最小化每代的训练时间。

        三级分层搜索算法：将 PCG 输入划分为多个子图，为每个子图选择优化后的替代序列和设备映射，再将这些子方案组合为最终的输出。

        - PCG 替代序列：Unity 使用 TASO 基于 cost 的回溯搜索算法来计算 PCG 的替代序列，以最小化执行时间。Unity 维护一个替代序列的队列（由执行时间排序），在空或满之前迭代移除最好的候选，并通过对其应用每个可行的替代来生成新的候选，若候选执行时间长于 thr 倍目前已知最优候选 PCG 的时间，则被剪枝；否则加入队列。
        - PCG 图分裂及设备映射优化：nity 通过序列图分裂和并行图分裂来递归分解线性和并行链为独立子图（直到无法分解），基于动态规划的思想。
            - 序列图分裂：在输入 PCG G 中找到一个支配节点 n（所有从输入到输出的路径都经过 n），并在 n 将 G 划分为两个独立子图 G_1 和 G_2（G_2 的 OP 等到 G_1 OP 全部完成才能开始）。该方法将找到 G 的最优设备映射规约为找到 G_1，G_2 的最优映射以及最优的 n；
            - 并行图分裂：将 PCG G 划分为可并行计算的独立子图 G_1 和 G_2，包含串行（全部硬件资源）和并行（共享硬件资源）两类运行方案，Unity 选择执行时间更短的哪个。为了在并行时划分可用资源，Unity 迭代遍历所有可能的资源量分配方案，忽略是仅 GPU 序号不同（而不是数量和位置不同）的方案，将指数搜索替换为二次搜索；

        相关 PCG 的映射信息存储在交叉调用缓存，因为存在替代关系的 PCG 间大部分图结构相同，可复用。

        为了解决扩展性的问题，Unity 将 PCG 分解为独立的串行子图，但这样会无法在子图间进行图替代，而 Unity 用替代来表示并行。因此这会将跨子图间的并行度降为 1，从而导致很多并行策略无法考虑。 

        对此，Unity 通过显式搜索子图间的最优并行化来解决该问题，必须保证前一个子图的输出 tensor 和后一个子图的输入 tensor 都符合考虑的划分，否则插入并行 OP 来调整，并考虑该操作带来的额外通信开销（即先遍历子图间 tensor 可能的合法划分，再根据该划分在两个 OP 内添加需要的并行 OP）。参考 MetaFlow，Unity 选择破坏最少替代的子图划分位置，并保持子图 size 不超过 k。这样做可以将最坏候选 PCG 数目从 g 的指数降为 g 的线性。

        Cost 评估：OP 运行时间和通信开销的评估与 OptCNN 和 FlexFlow 的方法一致。

        管道并行：使用 1F1B 调度和 weight update semantics，仅考虑应用在所有 OP 的管道并行（否则会产生瓶颈），每个 stage 仅和下一个 stage 通信，且假设 mini-batch 里的 micro-batch 比 pipeline stages 显著多，这样 pipeline 的初始延迟可被忽略。管道并行时的搜索算法直接以最大化吞吐为优化目标（而非以最小化每代运行时间）。

        整篇 paper 最 fancy 的地方在于更新了六大类并行策略的定义，提出并行计算图 PCG 将代数变换和并行策略统一表示为 OP，通过图替代的方式进行共优化，并通过三层分级搜索算法获取最优 PCG 替代序列及相应的底层设备映射。

    - Link: [Notes for Unity](https://github.com/DicardoX/Notes_for_Papers/tree/main/Unity)

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

