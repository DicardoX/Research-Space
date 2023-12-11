# Optimus: An Efficient Dynamic Resource Scheduler for Deep Learning Clusters

Optimus 是一个面向分布式训练 job 的调度器，主要目标是最小化 JCT，主要分为对 DL 模型收敛时间的预测，对模型训练速度的评估，以及一个考虑资源分配和 task（相较于 job 的更小级别）放置三个部分。

- **收敛时间的预测**：Optimus 对 DL 模型收敛时间的预测**基于 SGD 收敛速率为 O (1 / k) 的事实**（该结论对学习率的变化有要求），使用 **non-negative least squares (NNLS) 求解器来进行在线的拟合**。
- **性能评估模型的建立**：Optimus 建立了数学化分布式训练任务和资源的系统模型，并分析了包括 **worker 前向（与 bs 相关） & 后向（与 bs 无关）时间**，**数据传输时间**（部分模型大小 / 带宽），**parameter server (ps) 的参数更新时间**，以及**通信开销**（与 worker 和 ps 的数目成线性关系）在内的**总时间 T**。基于上述分析，Optimus 构造了一个表示**训练速度 f = 1 / T 的函数**，注意该函数**以 worker 和 ps 的数目为自变量**，对于同步和异步训练有些许区别（前者需要进一步考虑用户对 overall batch size M 的输入），并**以 NNLS 拟合的方式，通过预训练和在线收集训练数据建立和不断改进模型**。
- **资源分配**：最后，**基于任务剩余的代数以及上面得到的训练速度函数 f**，Optimus 定义了**边际收益**，来以**启发式（贪心）的方式进行资源分配**（不用整数规划是因为非线性甚至非凸，且 NP-hard）。
- **Task 放置**：实际上是**对 job 所被分配的 workers 和 ps 进行 servers 资源的分配**，分配完后**均衡地放在这些 servers 上**。Optimus 分析了**达到最小化最大数据传输的方式**，包括使用最少的 servers 来部署以及均衡地在这些 servers 上面放置 workers 和 ps，并以此提出了**基于贪心的分布式策略**。此外，为了解决 stragglers 的问题，Optimus 分别**对 workers 的训练速度进行均衡**，并提出了**对 ps 负载进行均衡**的方法。

整篇 paper 最 fancy 的地方在于**对 DL 模型收敛时间的建模和预测**，以及**对 workers 的训练速度和 ps 负载进行均衡**。然而，Optimus 对模型训练时间的预测在其他工作（Tiresias）中被证明为过分简化了损失函数的变化曲线，在实际集群中并不总是适用的。
