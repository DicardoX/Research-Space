# Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning

Petuum 提出的将分布式训练中的 **metric modeling** 和 **resource scheduling** 结合起来 **co-optimize 的分布式训练和调度架构**。

Pollux 可以同时用在 parameter server 和 all-reduce 两类分布式架构，原因是**重点关注 goodput 为主的性能指标**，而**非 ps 和 all-reduce 等架构中最重要的通信量的优化**（及参数同步的方式），这也是 Pollux 可以优化（考虑不足）的一个点。

考虑的主要 metrics 为 **statistical efficiency**（表征每一轮迭代模型精度能够进步多少） 和 **throughput**，并分别**对这两个 metric 的 estimate 进行了 modeling**。Pollux 使用 **gradient accumulation 来进一步扩大 batch size**，经过 s 轮的 GPU 本地梯度聚合后再进行全局的梯度聚合。基于这两个 metric，Pollux 提出了**自己的调度和优化 metric，即每个job 的 goodput**。整个 Pollux 的架构分为两层：

-  **Job 层面**：Job 指某个完整的 training model，每个 job 内部采用 data parallelism 等并行方式。**PolluxAgent 收集 bs 和 $T_{iter}$ 等信息**，基于信息**获取 efficiency 和拟合 throughput 的函数**，进而**获取每个 job 的 goodput 函数**；通过**最大化 job 的 goodput 来动态调整 bs 和 lr**，以更好地利用资源。最后，**周期地向 PolluxSched 报告 goodput 函数**，等待**新一轮资源分配后再调整 bs 和 lr**（lr 随 bs 线性变化）。需要注意的是，Pollux 采用 **online model fitting** 而非 profiling 的方式进行**在线的 thr 函数拟合**（使用之前所有的 thr 数据），再用 goodput = efficiency * thr 来获取拟合后的 goodput 函数。
-  **Cluster 层面**：**PolluxSched 基于 jobs 的 goodput 动态重分配资源**，通过**最大化 fitness 函数（由相较于 fair allocation 的 speedup 构造）来获取理论最优的分配**，并**考虑多个集群层面的目标**，包括 fairness，goodput，reallocataion penalty，interference slowdown 等。

注意，与以往工作（如 BytePS）不同的是，Pollux 不再是**在考虑集群通信拓扑和算力的前提下被动地适应**，而是**用 metric modeling 来为 resource allocation 提供依据，进而主动地共优化**。

整篇 paper 最 fancy 的地方在于它的建模过程，包括 **metric modeling** 和 **scheduling (optimization) modeling**，以及 **job-level 和 cluster-level 共优化的架构设计**。
