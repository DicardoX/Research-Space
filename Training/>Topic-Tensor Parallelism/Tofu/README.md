# (Tofu) Supporting Very Large Models using AutomaticDataflow Graph Partitioning

Tofu 和 OptCNN 及 FlexFlow 想解决的问题一样，是同时期**对自动化并行的探索**。相较于 layer-wise，Tofu **以 OP 级别的 tensor 为粒度**，将**大模型粗化后的数据流图**以 **partition-n-reduce** 的方式，**等分划分**到多个 GPU（仅划分各类 tensor，每个 GPU 都拷贝一份完整的图 OP），以**减少 GPU 的内存足迹**，同时达到**并行化**的效果；Tofu 使用一个简单的 Halide-like 语言 **TDL** 来**描述 OP 的语义**；在划分 OP 时，Tofu 使用一个 **DP 套递归的搜索算法**来**最小化通信开销**。

------

###### Part 1. Introduction

- 先前工作表明，从 80 年代开始最新 **NN 的参数量每过约 2.4 年就会翻倍**，这由硬件改进和大数据集的可用性支持；
- 两种模型并行的方式，一种是**将不同 layers 分配到不同设备**，一种是**将每个 layer 的 tensor 划分到不同设备**。**后者对大模型更好**，**每 GPU 内存使用率更均衡**，且对模型加速很重要；
- Tofu 可以划分 MXNet 数据流系统内各 OP 的输入/输出 tensors，被定义为 **OP 划分**，**比 layer 划分更为细粒度**（layer 可包含多个 OP 操作，如全连接就包含了 conv OP 和另外的 OP）；
- **Partition and reduce** 也包含小小的缺陷，其实 partition 之后，不见得”立刻“ unreduced，可以**让中间的 partial 结果在系统里继续参与计算**，这样反而有可能效率更优  ---- OneFlow。

--------

###### Part 2.  How to partition a single operator (TDL Description)

TDL (Tensor Description Language) 受 Halide 启发，核心 idea 是 **tensor as a lambda**，**从索引变量映射到值**，被表示为 **TDL 表示**，包含**索引变量**、**tensor 元素**、**算数 OP** 和 **tensor 沿一或多维度的规约方式** 。TDL **被实现为 Python 的 DSL**，**简单且非图灵完备**，例如不支持循环或递归。此时可表示为**模糊函数**（Opaque Function），有时含 batch 版本的模糊 OP 可沿 batch 维度划分。

**划分策略可通过描述每个 worker 进行各自份额计算所需的输入 tensor 区域来确定**。若 tensor shapes 已知，从 TDL 描述中获取输入区域是直接的。TDL **在抽象域中进行 range 分析**，即**符号中间分析**：将**索引变量 x_i 的 range 表示为 [0, X_i]**，然后**符号化执行 lambda 函数**来计算指定 OP 输入 tensors range 的符号中间表示。将**符号中间表示 I** 表示为**所有符号上界 X_i 的仿射变换**。Tofu 研究两类 tensor 划分策略（可表示为相应维度上的输入区域）：

- **Partition-n-reduce without reduction**：例如沿 b 维度划分，后续直接 concat，不需 reduction；
- **Partition-n-reduce with reduction**：例如沿 ci 或 dx 维度划分，结果直接进行 reduction。

---------

###### Part 3. How to partition the dataflow graph (Graph Coarsening & Recursive DP search)

**优化目标**：Tofu 选择**仅最小化总通信开销**，原因有两点：

- 大模型使用的 GPU kernels 处理大 tensor，因此**对不同维度的 tensor 划分执行时间相似**，因此**通信开销更低往往意味着更低的端到端执行时间**（意思是不管怎么划分，每个 GPU 的执行时间都基本相同，而且 Tofu 默认了划分到每个 group 仅一个 GPU，忽略了某些 GPU 要处理的 tensor 更大的情况）；

- 每个 GPU **用以存储 tensor 数据的内存总是等分的**（因为划分时都是等分的），而**用以 buffer  GPU 间通信数据的内存与通信量成比例**，因此**更低的通信开销会带来更小的 per-worker 内存开销**。

Tofu **粗化数据流图**，**将非线性图变换为线性图**，通过分组或合并多个 OP 或 tensors：

- **组合 fp 和 bp OP**：每个 fp OP 和自动生成的 bp OP 归为一组，每个 fp tensor（weight 或中间 tensor）和其梯度 tensor 归为一组。若权重 tensor 在 fp 时被多个 OP 使用，bp 时有多个梯度，链式法则要求相加，并将加法 OP 加入到组中；
- **合并 OP**：将**连续的 element-wise OPs 合并**。需要根据 TDL 描述来决定一个 OP 是否为 element-wise；**合并展开的时间戳**：RNN 不同的时间戳共享相同的计算逻辑和权重 tensors，因此也应该合并以共享相同的划分策略，这样多层 RNN 的数据流图变成了一系列合并且分组后的 OP。MXNet、Pytorch 等现有框架引入内建函数的概念，以将 RNN 的基本单元展开为许多时间戳。

**DP 套递归的优化算法**：

- 对一个**粗化后的图 G**，调用**完整的 DP 算法**（OptCNN's DP）将**已有 workers 等大小划分为两个 groups**，将**图 G 里的每个 OP 的相关 tensors 沿某个维度划分，并归于两个 groups**（每个 group 拷贝一份完整的图 OP），以上过程为一个**递归轮次**。不同 groups 间为了计算目标 sub-tensor 可能需要 **fetch extra data**。
- **划分后的子图 G_0 继续递归上述过程**（直到每个 group 仅一个 worker），并将最后的**划分结果应用到 G_1 上**。能直接应用的原因是 G_0 和 G_1 是同维度等分划分后的产物，具有高度相似性，这样可以节省一半的计算量。
    - 若 GPU 数目不是 2 的指数，则因数分解，并在每轮递归中划分到 k_i 个 workers 上（而非固定 2 个）

Tofu 理论上证明了**划分策略的最优性**，相当于**每次递归都选对通信开销增长最小的**。证明思路是不管怎样，最终的 seq 根据可交换性，交换为和最优 seq' 满足某两个连续划分策略前一个 seq 好，后一个 seq' 好，这时候更换当前的选择即可。同时，Tofu 的递归策略是**多级带宽友好**的。递归过程中，整体通信开销逐渐增大。由于 group 规模从大到小，仅开始阶段需要在集群顶部带宽较小的区域进行数据传输；随着递归的进行，Tofu 仅在同一个 server 内（高带宽）进行某一轮的 DP，因此受带宽限制较小。

此外，Tofu 还提出**利用框架已有的内存管理器**：数据流图的划分会**改变原来 OP 间的依赖关系**，因而**阻碍 OP 间的内存 buffer 再利用**，导致 per-worker 内存开销更大。Tofu 根据原数据流图，**在划分图中生成额外控制依赖关系**，来解决上述问题。对原图中的 OP，每个 GPU worker 在划分子图中生成一个拷贝，**通常需要从其他 workers fetch data**，这会产生许多中间内存块，导致**更大的 per-worker memory**。为了缓解上述情况，我们引入一个**自定义 GPU kernel，MultiFetch**，使用 **CUDA Unified Virtual Addressing (UVA) 来直接访问另一个 GPU 的内存**，**避免 kernel 执行前的显式数据拷贝**。MultiFetch 获取其他 GPUs 输入区域内存块的指针，并在一次 kernel launch 中集中，以便后续执行 data fetch 时利用这些指针直接访问其他 GPU 的内存。

----------

###### Part 4. Limitations

- Tofu **仅研究了整个数据流图 G 的 OP 数据/权重 tensor 划分**，但**并不支持将不同 OP 划分到不同 GPU 上**（模型并行的一种，但 part 1 提供了解释），但**支持数据并行**（即对输入 tensor 沿 batch 维度划分）；
- Tofu **假设了迭代优化时必须重复到每个 group 仅对应一个 worker**，基于**更多的并行化也会带来更低通信开销**的事实，这点设计是合理的。然而，**当模型较小时**（相较于可用 GPU 数目），**降低通信开销的程度有限**，且**单 GPU 上的利用率也会不足**，这种情况下保持部分 OP 不划分或部分划分可能会更好，占用的硬件资源将远多于实际需要使用的硬件资源（换句话说，提供多少可用资源，设置多少递归轮数，需要管理者根据模型大小自行决定，即**没有设计相应的资源管理和分配策略**，虽然不是关注点，但仍不完整）；
- Tofu **没有考虑 GPU 间的异构性**，不支持非等分的划分，即没有细致地考虑硬件 profile 和分配问题，仅仅是看作 2^m 个完全相同的 workers；
- Tofu **仅支持 partition-n-reduce 的并行**，**限制了每个 worker 进行与原图相同的粗粒度任务**，且该策略**不一定能使通信最小化**，也**没有利用底层的互联拓扑**（尽管搜索算法尝试去适应多级拓扑的带宽差异，它仍然没有显式优化互联拓扑间的通信）。
- TDL 的局限性：1) TDL 不包含 flow primitives 和 data-dependent indexing；2) 由于负载不均衡，Tofu 不支持 sparse tensor operations，尽管有相关 TDL 描述；3) Tofu 不验证 OP 实现是否与其 TDL 描述匹配 ；

--------

###### Part 5. Conclusion

整篇 paper 最 fancy 的地方在于**设计了 OP 语义的描述 Python DSL TDL**，**以 OP 为粒度进行 tensor 的多维度划分**（这很好地适应了 Pytorch 等现有框架），提出了**基于 OptCNN DP 套递归的搜索算法**（由 partition-n-reduce 等分性直接对半缩减计算量），并针对 GPU worker 需要跨 GPU fetch data 提出了利用 **CUDA Unified Virtual Addressing (UVA) 来直接访问另一个 GPU 的内存**，**避免 kernel 执行前的显式数据拷贝**。可惜的是，由于 Tofu 底层的 DP 算法不能为未划分或未等分划分的 OP 进行设备放置（**设备放置的决策**同**根据 worker group 递归搜索划分**相绑定）的优化，Tofu 的递归搜索不能被扩展以解决上述限制。

