# Tiresias: A GPU Cluster Manager for Distributed Deep Learning

Tiresias 是一个面向参数服务器（Parameter Server）架构下分布式训练 job 的调度器，主要包括一个 2DAS 调度器，以及基于模型结构对合并放置约束的放宽策略。

首先，Tiresias 讨论了**同时考虑时间（JCT）和空间（资源分配）两个维度（2D）以计算优先级的重要性**，以及**优先级离散化的优势**。Tiresias 进行架构设计的前提条件包括：**DL job 的执行时间通常是不可预测的**，**job 的特点无法提前获知**，以及**同步 PS 架构下 DL job 的资源分配具有 all-or-nothing 的特点**。

- **2DAS 调度器**：当**无先验信息**时，优先级函数采用 **2D-LAS 算法**，**job 获取的资源量等于当前执行时间 * 分配资源量**；若**提供部分先验信息**（比如执行几轮后，但 paper 中并没有说明如何通过几轮的信息拟合得到 JCT 分布，只说需要管理者提供该分布），**优先级的值则与 Gittins index value 相等**，Gittins index 值代表**该 job 获取一定量服务后在拿到下一个服务量后能完成的可能性**，值越高表示优先级越高。
- **优先级离散化**：参考**多级反馈队列（MLFQ）的架构**进行设计。**LAS** 时，**对相同队列里的任务采用基于 start time 的 FIFO**；**Gittins index** 中，service quantum 被设置为当前队列的服务量上界，**当 job 消耗完 \Delta 时被降级**。**相同队列中的 jobs 以各自的 Gittins index 值来调度**，最低队列（\Delta 为无穷，无法计算 Gittins index value）以 FIFO 的方式。
- **合并放置约束**：Tiresias 使用**模型结构的倾斜程度来预测 jobs 是否对合并放置敏感**。部分模型的某些层是很大的 tensors，而在模型聚合时该层参数的信息大小与 tensors 大小相关。因此，**聚合大 tensors 更容易受到网络竞争的影响**。由于每个 parameter server 都会周期性地向每个 worker 发出自己拥有的更新后的模型参数，Tiresias **通过监控网络通信来获取模型倾斜信息**，并构造了一个自己的监控工具。

整篇 paper 最 fancy 的地方在于**面向无 JCT 分布信息和部分信息抢占式 2D 调度算法设计**，**参考多级反馈队列设计的优先级离散化架构**，以及**基于网络通信对模型 tensors 倾斜信息的监控方法**。
