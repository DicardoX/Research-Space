# ElasticFlow: An Elastic Serverless Training Platform for Distributed Deep Learning

本文提出了 **ElasticFlow**，一个面向分布式训练的弹性无服务训练平台，支持 user **仅指定 model，超参** (not GPU num) 和 **job DDL** (not GPU time)，从而**在提供满足 DLL 性能保证的同时隐藏底层资源管理**。
分布式训练的两个挑战包括：(1) 训练吞吐随 GPU num 非线性增长 (**non-linear scaling curve**)；(2) Worker placement 影响 scale 效率 (**Topology-aware**)。为了解决上述挑战，提出 **Minimum Satisfactory Share** 来判断**满足 job DDL 所需的最少资源量**，并基于此进行**准入控制 Admission Control** (判断 job admit or drop)。ElasticFlow 还设计了一个**贪心算法**，来**基于 marginal return 设置优先级动态分配资源**，在 worker placement 上使用 **Buddy Allocation + Job migration 来消除拓扑的影响**。

------



### 1. Background

- ElasticFlow 的关注点不在加速训练 job，而在于多任务场景下的资源调度，以利用弹性特性并以无服务的形式提供性能保证。
- Server-centric model 将底层系统问题和 DL 问题绑定，而 Serverless 将二者解耦。
- 大多数已有工作关注优化 JCT，但还有另一种场景是开发者根据自己 job 的预期 DDL 来请求性能保证。
- **尽管最近有工作尝试考虑 DDL，但仍然用的是 server-centric 方法，缺少弹性 scale 资源以优化集群资源利用并满足 DDL 的灵活性**。

-----



### 2. Architecture

- 将训练 job encode 成一个单机训练的函数，只需指定 global bs 和其他超参，系统级别的资源管理问题 (基于 GPU memory 决定 local bs 和 workers 数目) 由 ElasticFlow 进行。
-  用户只需指定 job DDL，无需考虑资源释放，这也方便了系统来基于 DDL 和终止条件动态调度资源。
- 用户可以通过多种方式指定终止条件，ElasticFlow 使用最大迭代轮数来作为主要的终止条件，也支持不带 DDL jobs 的调度。
- Incoming jobs 的两种状态：admit or drop。
    - <font color=blue>为啥不 queue? 如何判断该 queue 还是该丢弃？即判断在未来是否可能存在一定的资源量来满足该 job 的 DDL 需求。不存在这种可能，因为求解的时候已经考虑了所有情况，如果 Job A 一直占用资源，Job B 等 Job A 结束后就能在 DDL_B 前完成，则求解器一定能考虑到这种情况，进而对 Job B 进行 admit.</font>
- **性能保证**：若 job 被系统接收，则其 DDL 被保证。

-----



### 3. Admission Control

- **资源用量的定义**：**GPU 数目 * GPU 占用时间**。由于收益递减效应的存在 (per-GPU 吞吐随 GPU 数目增多而下降)，单 GPU 的资源用量最低，但有时候可能会违反 DDL 需求。

- **Minimum Satisfactory Share：可以满足 job DDL 的最少资源量** (注意“资源量” (e.g., N 块 GPU) 与“资源用量”的定义区分)，MSS 能使得 job 的资源用量最小化。

- 当集群中没有其他 jobs 时，可以直接用二分查找来获得满足 DDL 需求的最少 GPUs。当存在其他 jobs 时，某个 job 完成所需的 GPU time 依赖于其他 jobs 的资源占用情况。

- (假设 linear scaling curves) 定理：对 jobs 按 DDL 升序排序，若任意 i，前 i 个 jobs 的 GPU time 之和小于 GPU 总数 * 第 i 个 job 的 DDL，则存在合法资源分配。

    - <font color=blue>这种 “1 个式子表示 n 个条件”类型的约束条件挺少见，注意。</font>

    <img src="./figures/Screenshot 2023-02-25 at 22.59.07.png" alt="avatar" style="zoom:50%;" />

    - 约束一：每个 job_i 均能在 DDL 前达到终止条件；
    - 约束二：任何时间 t，所有 jobs 分配的 GPU num 不超过总 GPU num

    <img src="./figures/Screenshot 2023-02-25 at 23.00.02.png" alt="avatar" style="zoom:50%;" />

- **渐进式填充算法 (Prograssive Filling)**：针对 non-linear scaling curves，使用渐进式填充算法，同样 jobs 按 DDL 升序排序，但渐进地增加每个 job 的 GPU 数目，直到 job DDL 被满足。

    - 每次需要 reschedule 时执行 admission control 判断，先把新来的 job 加到 jobs 队列，**按 DDL 升序排序，再按顺序对每个 job 调用渐进填充算法**，看该 job 是否能被满足 DDL。**若存在某个 job 不能满足，则 drop 新来的 job**。上述算法**本质上是一种贪心算法**。
    - 注意，所有 jobs 都开始时间均为 t=0,当新 job 来时，计算已有 jobs 的剩余 iterations num，并将开始时间全部置 0。<font color=blue>每次重调度如何利用 jobs 之前 run 得到的先验信息？</font>

------



### 4. Resource Allocation

- **资源分配策略的目标**：**在 minimum satisfactory share 的基础上，将空闲资源分配给 admitted jobs**。两类直观的方法：(1) 将空闲 GPUs 全部分配给 DDL 最早的那个 job，但存在收益递减效应；(2) 将空闲资源均匀地分配给 admitted jobs，但未考虑每个 job 已分配的 GPU 数目。
- **剩余资源优化问题的核心**：**给定每个 job 已分配的 GPU 数目，考虑收益递减效应** (分配越多 GPU num 给一个 job，则其消耗的 GPU time 越多)。
- **优化目标：最小化全部 jobs 的总 GPU time**。
    - 约束包括三个：(1) 所有 jobs 的 DDL 被满足；(2) 总分配 GPU 数目不超过当前可用 GPU 数目；(3) **在下一个 time slot 中所有 GPUs 均被分配，除非给任一 job 分配更多资源都会导致执行更慢**。 

----



### 5. Job Placement

- **跟 HiveD 基本相同**。
- 重要性：**Job 的 scaling curve 依赖于 placement，并直接决定了准入控制和资源分配的准确性**，进而影响 job DDL 的满足保证。
- 一个简单的方法是**总是使用最坏情况下的 scaling curve** (job 的所有 workers 在不同机器上)，这会低估 job 吞吐，进而高估资源用量 (GPU num * GPU 占用时间)，从而保留不必要多的 GPUs，导致 admit 的 jobs 数目变少。
- 使用该方法来建模 GPU 拓扑，规律的进行 GPU 分配。
    - <font color=blue> 进而预设多种不同子树下的 job scaling curve (并没有这么干) </font>

- **Topo-aware Job Placement**：参考 HiveD 使用**多级架构建模集群**，使用 **Best-fit 来将包含与所需 GPU 数目最接近的子树分配给 job**，从而**使其获得尽可能高的带宽拓扑**，避免吞吐的低估和资源用量的高估。同时，ElasticFlow **应用 buddy allocation + job migration 来消除 resource fragmentation**.

-----



### 6. Discussion

- **Non-DDL jobs**：对 best-effort jobs 设置 DDL 为无穷，并在为所有 SLO jobs 分配 minimum satisfactory share 资源后将剩余资源全部分配给 best-effort jobs (在满足 DDL 的前提下，最小化 best-effort jobs 的 JCT)。**实验结果表明 DDL 达标率较好，但 best-effort jobs 的 JCT 提升不如其他工作**。
- **恶意用户**：我们假设用户不是恶意的，并根据他们的需求来设定工作期限。可以使用 quotas 和额外收费来限制。
