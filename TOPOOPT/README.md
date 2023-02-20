# TOPOOPT: Co-optimizing Network Topology and Parallelization Strategy for Distributed Training Jobs

TOPOOPT 是一个**面向 DNN 训练负载的直连架构**，在**计算、通信和网络拓扑三个维度共优化分布式训练**过程 (即共优化网络拓扑和并行策略)。本文证明了 **AllReduce 流量的可变性**，以此构建 **DNN 训练 jobs 的高效网络拓扑**。TOPOOPT 使用一个**优化技术**和 **TotientPerms (group theory-based) 算法**，来**发现最佳网络拓扑，routing plan 和并行策略**。

可以 cite 它说它考虑的是 Swith - server 两级架构，而大多数产业集群为多级架构，情况更复杂。

###### Motivation

**查找最优网络拓扑很困难**，必须同时满足：(1) 高效的大规模 AllReduce 传输；(2) 保证较少的模型并行传输的跳数 (**这里的 MP 指某些 layers 在 nodes 间切分，而非某台 server 内部，目的是为了减少 AllReduce 的通信开销**)。因此提出 TotientPerms，一个 group theory-based 技术，来利用 AllReduce 的流量可变性，构建一系列 AllReduce 组合，在高效完成 AllReduce 传输的同时合理放置以进行模型并行传输，进而提高整体训练性能。

先前工作未考虑物理层拓扑作为优化维度。

###### Design

我们将网络拓扑和并行策略的共优化问题表述为一个离线的选择性优化框架，可在优化并行策略和优化网络拓扑间选择。首先假设 fixed 拓扑来在并行策略空间中搜索，并将 traffic demand  返回给拓扑发现算法；更新拓扑后，将重新返回到并行策略搜索算法。

d 个 switches，每个 server 分别与每个 switch相连，支持将集群划分为每个训练 job 专用的分区 (size 依赖于资源需求)。TOPOOPT 首先离线地查找 servers 间的最优并行策略和拓扑，再重配置 switches 来实现当前 job 的目标拓扑。

Server 的度一般小于 server 需要通信的邻居，为了保证两台没有直连的 servers 间的非阻塞通信，使用 host-based forwarding，即让 hosts 像 switches 那样向 destination 转发流量。

###### Algorithm

将**搜索空间分为 Comp * Comm (并行策略 * 通信开销) 和 Comm * Topo (通信开销 * 拓扑) 两层**，使用选**择优化来迭代地搜索其中一层，并保持另一层不变**。
具体来说，使用 FlexFlow 的 MCMC 搜索算法来在固定网络拓扑下考虑通信开销地寻找最佳并行策略，若提升则将结果返回给 Comm * Topo 层；再调用拓扑搜索算法来找到当前并行策略下的最优拓扑和 routing。并再返回给 Comp * Comm 层来进一步优化并行策略和设备放置，直到收敛或 k 轮。

**算法步骤**：

- Step 1. 根据 traffic share 按比例为 AllReduce (DP) 和 MP 的子拓扑划分连接端口 (即度 d).
- Step 2. 为了获取 AllReduce 子拓扑，算法根据环内每个 group k 的 traffic amount 来成比例地划分连接端口，并使得集群的直径 (集群内所有 ring (一个 ring 最小可以对应一个 layer) replicas 数目，通过每个 replica 连接多个邻居，这样就能保证单个连接通信量不变的情况下，每个 replica 尽管需要同步的 weights 更多，但整体速度和更多 replicas 的情况保持一致) 最小，具体包括 TotientPerms 和 SelectPermutations 两个算法
- Step 3. 为了获取 MP 子拓扑，使用 Blossom 最大权重匹配算法来根据并行策略找到子拓扑；
- Step 4. 将两类子拓扑结合，AllReduce 用修改版 coin-change 算法来在 AllReduce 子拓扑内 route，MP 用 k-shortest path 算法在 MP 子拓扑内 route。

###### Traffic Mutability

**传统方法**将 Comp * Comm 层的并行策略和设备放置映射为一个 traffic 矩阵，并将矩阵映射为 circuit 调度。但在分布式训练中会带来问题：AllReduce transfers 比 MP transfers 更大，但后者通信度数更多。因此，传统方法会导致尽可能为 AllReduce 提供更多的平行直连，而 MP flows 的跳数会很大，这会导致训练性能的降低。

**可变性**：在不改变并行策略和设备放置和保证正确性的前提下，改变 traffic pattern。由于 MP 各设备上的模型部分不同，AllReduce 相同，因此仅后者有可变性。

**TOPOOPT 的策略**：(1) 为 AllReduce transfers 分配足够带宽；(2) 保证 MP transfers 的跳数较小。
通过利用 AllReduce 的可变性来实现。

**核心思想**：由于 AllReduce 的可变性，不同排列可以有相同的 AllReduce latency，但可以有不同的 MP transfers 跳数，并加上省去不同 AllReduce 排列重叠部分的搜索，进而高效地查找 AllReduce + MP 的拓扑。

**TotientPerms 算法**：仅考虑全部 n! 个 AllReduce 排列中的 regular rings，即 node_i 固定与 node_{i+p} 相连，对任意 i，p 满足为素数且和 n 互质。

**排列选择算法**：对每个 group (ring)，基于 TotientPerms 算法的结果，选择 d_k 个排列，应用到每个 group 中，从而仅可能减少集群所有 ring 的半径，进而有利于 MP transfers。

- d_k 为每个 group (ring) 内各节点的度数 (相同 group 内节点度数均相等)。



