# Lyra: Elastic Scheduling for Deep Learning Clusters

Lyra 是一个跨 training cluster 和 inference cluster 的协同调度器，通过动态借用 infernece cluster 内的闲置机器 (Capacity Loaning)，并弹性调整 training cluster 内的任务资源量 (Elastic Scaling) 以最小化 JCT，进而解决两个问题：(1) 低负载导致的推理集群利用率低；(2) 训练集群缺少资源导致排队时间长。

具体来说，Lyra 负责接收由 inference cluster scheduler 生成的闲置机器的信息 (非 Lyra's design，不会影响 inference jobs)，通过 elastic scaling 在 training cluster 内用起来 (通过后续的 resource allocation 策略)，并在 inference cluster 负载升高时将借用的机器归还 (Reclaiming，通过定义 Server Preemption Cost 最小化 job preemptions)。此外，Lyra 还制定了 resource allocation (two-phase heuristic，先是 SJF for inelastic jobs，然后是 multiple-choice knapsack problem for elastic jobs) 和 job placement (bin packing with best-fit de-
creasing (BFD) heuristic) 的策略。



