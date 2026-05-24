# (BytePS) A Unified Architecture for Accelerating Distributed DNN Training in Heterogeneous GPU/CPU Clusters

ByteDance 提出的将 PS 和 Ring All-reduce 两种架构综合考虑的一种**统一集群内通信架构**，利用**集群中空闲的 CPU 和带宽资源**，并将 PS 和 Ring All-reduce 成功表述为统一架构下的特殊情况。

BytePS 抽象了两类服务：

- **SS (Summation Service)**：只**运行在 CPU 中**（包括 CPU 机器和 GPU 机器），负责**从 CS 侧接收 tensors**，把 tensors **加**起来，再**返回给 CS**；
- **CS (Communication Service)**：只**运行在 GPU 中**，负责**同步多个局部 GPU 之间的 tensors**。

BytePS 使 **CPU 仅负责 sum（SS）**，而 **GPU 负责 FP、BP 和 parameter update**。该架构对 CPU 和 GPU 各自所需的通信时间进行建模，令 t_c = t_g 确定最优通信时间 t（关于 size M、带宽 B、GPU 数目 n 和 CPU 数目 k），进而能够**根据集群中可用 CPU 和 GPU 的相对数目**，动态决定 **SS 中分配在 CPU / GPU 的数据的比例**，这样得到的机器间通信是**延迟最优**的。对于两类模块的通信量建模基于以下几个假设：

- GPU 机器上的 CS 模块需要收发 $M - M_{SS_{gpu}}$ 大小的 bytes，SS 模块需要从其他 n - 1 个 GPU 机器各自收发 $M_{SS_{gpu}}$ 个 bytes（**即该份模型参数使用 ring allreduce 进行同步**）；
- CPU 机器上的 SS 模块需要收发 $M_{SS_{cpu}}$个 bytes（**即该份模型参数使用 ps 进行同步**）；
- $M = k \times M_{SS_{cpu}} + n \times M_{SS_{gpu}}$；
- t = M / B

**由于 GPU 和 GPU、GPU 和 CPU 之间使用独立的带宽，因此这两类通信并不会互相干扰**。进一步，可以用**加速比**的概念和 allreduce 及 ps 策略进行比较。

同时，BytePS 研究了**机器内拓扑结构对通信效率的影响**，通过 **profile 获取拓扑**，针对 **PCIe-only** 拓扑提出了 **CPU-assisted aggregation 策略**（可 pipeline），针对 **NVLink-based** 拓扑使用 **reduce and broadcast 策略**。

由于在 **sum 之前 GPU 就要 update 参数**（即 CPU 不再进行乘法操作），因此破坏了 PS 原有对异步并行的支持性， BytePS 因此提出**支持异步的参数更新算法**，即向 CPU 传输 delta w_{t}（已经进行过乘法操作，而非梯度 g_t），并证明了其和 PS 异步并行的等效性。

整篇 paper 最 fancy 的地方在于**对加法模块和通信模块的功能定义**和**对通信模型的建立**，较为详细且严谨；以及对不同机器内拓扑下基于 profile 的数学建模和策略设计。
