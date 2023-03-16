# Gandiva: Introspective Cluster Scheduling for Deep Learning

Gandiva 是一个**集群调度框架**，通过利用 profile 信息来优化延迟，提高训练效率和集群利用率。考虑如下 DL 特点：1) **Feedback-driven exploration -> simultaneous early feedback**；2) DL jobs 在资源利用上的**异构性 (locality & interference)**；3) **Intra-job predictability**。Gandiva 基于以上特点，设计 **suspend-resume** (time-slice with weighted round robin), **pack** (对应单 GPU 混布多任务), **migration** (去碎片化), **grow-shrink** (提高集群利用率 & 为新 job 提供资源以共优化) 等机制，通过 **profile 监控：1) 资源使用情况；2) job progress rate (mini-batch time)** 来判断调度策略的有效性。在**调度策略**上 (**reactive mode -> 事件处理**，**introspective mode -> 持续监听并优化 job replacement**)，Gandiva 倾向于将 job 分配到相同 **affinity** (相同需占 GPU 数目) 的 server 上，并通过 **over-subscription** (time-slice & suspend-resume) 来尽可能消除新 job 的排队时间，**保证 early feedback**；基于 **profile 和一个贪心启发式算法**，利用 **packing, migration, grow-shrink** 等方法来 **runtime 优化 cluster efficiency** (包括 util)。注意，Cluster fairness (servers 间 load 均衡) 并不考虑，仅考虑 server 内 job fairness (time-slice weighted RR 来避免 job 长时间等待)。

-------



## 1. Job Migration

这部分在 Gandiva 中给出了较为详细的分析。

- 当 job departures 时触发，用来提高 locality。

- 修改 TF 源码，实现了一个 Migration Helper on each node，来进行 migration 的各类操作。由于是 DP，因此不把 meta-graph 包括在 checkpoint 内。

    - 使用 Ramdisk 将 checkpoint 保存在内存中，以加速迁移；当跨 server 时，通过 NFS 协议使用 remote Ramdisk。

- 跨 server 迁移实验：ResNet-50 job

    - 怎么测的？

    <img src="./figures/Screenshot 2023-03-16 at 15.09.29.png" alt="avatar" style="zoom:40%;" />

- Gandiva 可以消除或隐藏 Migration overhead

- 实际的 migration time，checkpoint 保存/恢复的时间基本保持 constant，无论 GPU 数目的多少。原因是 Gandiva 仅使用数据并行，所有 GPU 上保存的模型参数量相等。

    - **当考虑 DP 时，由于参数量不同，因此仅存在上界**（仅包含部分参数的 worker 如何实现 job migration？直接重启吗？需要重启其他 workers 吗？）

- 读取 checkpoint 时每个 GPU 并行读取，不会造成太大的 PCIe 带宽压力

- 8-GPU job 的迁移约花费 35s，1-GPU job 的迁移为秒级（**分布式训练中单个 worker 的迁移会更快**）

- **Intra-server migration 普遍比 Inter-server migration 要显著更快**。

    <img src="./figures/Screenshot 2023-03-16 at 15.58.20.png" alt="avatar" style="zoom:50%;" />

- Gandiva 进行 job suspend 时最多 delay 一个 mini-batch time，以保证最少的 GPU-CPU 拷贝开销

- 

