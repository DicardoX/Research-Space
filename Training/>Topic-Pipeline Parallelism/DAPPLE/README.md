# DAPPLE: A Pipelined Data Parallel Approach for Training Large Models

DAPPLE 是一个**同步训练框架**，将大模型的**数据并行**（stage-level replica）和**流水线并行**统一，在保证训练收敛性的同时提高内存效率。DAPPLE 由 **DAPPLE profiler**，**DAPPLE planner** 和 **DAPPLE runtime** 组成。**Planner** 尝试**求解 stage 划分，replica 数目和设备放置问题**，探索数据和流水线并行的最优混合策略；**runtime** 包括一个**基于依赖关系的 early backward scheduling & warmup 调度算法**，在**减少设备内存使用**的同时保证**不影响吞吐**。实验结果表明 DAPPLE planner 在同步训练场景中相比 PipeDream 获取了 3.23x 的加速；runtime 则比 GPipe 节约了 12% 的内存，同时获得了 1.6x 的训练吞吐。

-----

###### Part 1. Introduction

- 在**异步 pipeline 并行**方面，尽管 PipeDream 等技术已经在提高 time-to-acc 方面取得提高，但异步训练由于**收敛性问题**仍应用不多。同时，异步方法需要**存储多版本的模型参数，占用更多内存**。对于**同步 pipeline 训练**，现有方法的**内存开销较大**，因为直到所有 micro-batches 的 fp 完成后才能 bp，一些重计算方法也会带来**额外重计算开销**；
- 面临的两个挑战：
    1. 如何在给定**模型结构**和**硬件配置**时进行**最优并行策略的决策**，注意目标优化空间包括 DP，pipeline 并行和二者的混合。现有方法中，PipeDream 无法有效应用到同步训练中，而 GPipe 需要经验性的人工优化，且未考虑一些并行维度；
    2. 如何**调度 pipeline stage 计算**，以达到**并行，内存开销和执行效率的平衡**。

-------

###### Part 2. Architecture

DAPPLE 主要由三个部分构成：

1) **Profiler**：（离线，秒级）将**模型作为输入**，获取**每个 layer 的执行时间，激励 size 和 参数 size**；
2) **Planner**：（离线，秒级）将 **profile 结果，硬件配置信息和给定的全局 bs 作为输入**，生成一个**优化后的混合并行方案**；
3) **Runtime**：采用**上述方案**，将**原模型图转换为 pipelined 并行图**，并将**全局 bs 划分为多个 micro-batches 并调度**（实现中，DAPPLE 并未实现图转换的自动化，而是人工转换）；

对于 DP 和 pipeline 并行的混合，DAPPLE 通过 stage 在多设备上 replica 实现，且可以**很好地适应多级带宽**（DP 需要梯度同步，stages 间需要激励通信）。

-------

###### Part 3. Planner

对于同步训练，将单个全局 batch 的执行时间作为性能指标，称为 **pipeline 延迟 L**。优化目标是在**同时考虑 DP 和 PP 的情况下最小化 L**。L 的计算主要由**最少 bubble overhead 的 stage（pivot stage Q）影响**，pivot stage 可能不是最后一个 stage。注意，将 **inter-stage 通信考虑为独立的 stage**，F_s 和 B_s 为其 fp 和 bp 过程中的通信时间。

Pipeline 训练 iter 由三个阶段组成：

1. **预热阶段**：从**调度开始**到**第一个 micro-batch 的 fp 执行完 pivot stage** 结束，对应 $T_w = \sum_{s=0}^Q F_s$；
2. **稳态阶段**：预热阶段结束到结束阶段开始，对应 $T_s = (M - 1) \times (F_Q + B_Q)$（实际上，pivot stage Q 也会包含部分 bubbles，但这里的建模没有考虑）；
3. **结束阶段**：从**最后一个 micro-batch 的 bp 执行完 pivot stage** 开始，到**全部任务完成**结束（包括每个 stage 内部可能需要的梯度聚合同步时间，allreduce），对应 $T_e = max_{s=0}^{S-1}(AR(P_s, g_s) + K)$，其中 K 是bp 时间的和（若 s > Q 则为负），AR 是每个 stage allreduce 的时间开销。

DAPPLE 有三类设备分配方法：

1) **Fresh First**：**优先将一个 stage 的 tasks 全部分配在相同的新 machine 上**，可以利用告诉 NVLink 进行 intra-stage 通信，但会导致碎片化（如果一个 stage 无法占满 machine）；
2) **Append First**：**优先将 stage 的 tasks 分配到已经有 GPU 被占用的 machine 上**，可以减少碎片化，且较大程度上可以将 stage 分配在相同 machine 上；
3) **Scatter First**：**优先将 stage 的 tasks 平均分配到所有已用 machines 上的空闲 GPUs**（如果全是新的 machines 则也是均分），适用于权重 size 相较于激励 size 可忽略的情况，能最大程度上减少碎片化。

**当前状态下 Q 的选择依赖于启发式算法，通过从 stage S-1 到 stage 0 挨个比较来确定 Q**。T^s\_{st} 是 stage Q 等待 stage s 执行的时间，T^{Q}\_{st} + \sum\_{s+1}^{Q-1} (F + B) 是 stage s 等待 stage Q 到 s+1 执行的时间，二者都还需要等共同的前面和后面的 stages，但因为相同所以不计入等待时间。等待时间即 bubbles。

**面向 stage 划分和 DP 的搜索算法**：

- 递归过程中，前 j 个 layers 已经划分为多个 stages，总共分配 m 个 GPUs，**后面的 layers 构成单个 stage s' 并 replicate 到剩余 G-m 个 GPUs 上**，此时**通过计算决定 pivot stage Q（用来计算 T_{PL}）**，并记录；
- 下一步，在 **stage s' 中的多个 layers 中再递归划分一次**，并分别将 s_1'（本轮考虑的 stages 构成）replicate 到 m' 个 GPUs 上（**枚举 m'，并利用三类设备分配策略决定具体放置**）， **s_2'（剩余 stages 构成）replicate 到剩余的 (G-m-m') 个 GPUs 上，利用 Q_j 辅助再次计算当前的 pivot stage Q'**（计算新的 T_{PL}），并记录；
- 最终选择**所有记录中 T_{PL} 最小的划分方案**。

几点 insights：

- **将 model 划分为尽可能少的 stages**，来**最少化相同 micro-batches 数目下的 bubble 开销**；
- **以较不均衡的方式划分模型的 stages**，可以获得更低的 bubble 开销和更好的性能，**本质上等于增大 stages 间的并行重叠**，stage id 越大分配的计算量应该越小。

-------

###### Part 4. Runtime

基于 TF 实现 DAPPLE runtime，以 model 和 planning res 为输入，将模型图转换为 pipelined 并行图。主要有三个步骤：1) **为每个 stage 分别构造 fp/bp 图**；2) **添加额外的 split/concat 节点**；3) **构造子图来在 sync 训练中更新权重**。注意，上述过程 **DAPPLE 需要人工进行**，未来可以自动化。

DAPPLE 引入 **Split-concat OP** 来实现**快速的 stages 间通信**，面向场景是**通过 DP 将同一 stage 的不同 replicas 放在不同 GPUs 上**，通过**并行缩短 micro-batch 的处理时间**，**以便下一 stage 能更快获得激励结果**。Replicas 之间的梯度聚合包含梯度累积步骤，通过 allreduce 同步后再各自更新权重。另一类方法是不 split，而是**将整个 micro-batch 以 RR 的方式在多 GPUs 上处理**（实际上每个 micro-batch 只在一个 GPU 上），这会带来更为**显著的尾延迟效应**，进而造成 **pipeline 效率低下**。

注意，DAPPLE 里并发的理解是同时注入多个 micro-batch，然后按 RR 的方式调度处理。

**Early backward scheduling** 是 **micro-batch 级别并行**和**峰值内存开销**的 **tradeoff**，相较于一次性插入所有 M 个 micro-batches，DAPPLE **在开始阶段先插入 K 个来在释放内存压力的同时保持较高的 pipeline 效率**；然后，我们在**各个 GPU 上严格调度一个 fp 后面接一个 bp**。这样，DAPPLE 在各 GPU 上实时保存的激励数目会明显少于 GPipe，进而在不损失训练效率的同时减少内存开销。注意，**K_i 是 stage i 在开始阶段需要调度的 micro-batches 数目，表征 stage i 的峰值内存开销**。

有两类策略来确定 K_i：

- K_i = min(S-i, D)，当 **ACR（activation communication ratio）较小**，**cross-stage 的通信开销可忽略**；
- K_i = min(2 * (S-i) - 1, D)，当 **ACR 较大**，**cross-stage 的通信开销与 fp/bp 计算开销相当**，此时用更多 micro-batches 来强化 pipeline 的效果。

实际上，stages 数目 S 要远少于 micro-batches 数目 M，这对限制峰值内存开销很重要，且 DAPPLE 鼓励更大粒度的 stage 计算以提高效率（uneven pipeline）。

--------

###### Part 5. Conclusion

整篇 paper 最 fancy 的地方在于**将流水线并行和数据并行结合**，**利用 pivot stage 的概念计算 pipeline latency 并划分 pipeline 阶段**，**面向 stage 划分，stage-level replication 以及设备分配的搜索算法**，以及 **early backward scheduling & warmup 的 runtime 调度算法**。
