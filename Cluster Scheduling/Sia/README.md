# Sia: Heterogeneity-aware, goodput-optimized ML-cluster scheduling

Sia 是一个面向异构集群的多任务调度器，同时考虑了 GPU num 的弹性 (*Adaptivity-aware*) 和 GPU type 的可变性 (*Heterogeneity-aware*)，以及任务训练超参 (batch size, learning rate) 的相应调整。Sia 是基于 Pollex 进行设计的，整体的优化目标是最大化 cluster-wide goodput，与 Gavel 类似地使用求解器来求解一个 ILP 问题，在该问题中枚举所有的 GPU num + GPU type 组合。此外，虽然 Sia 声称可以面向 hybrid parallelism，但其主体设计仍是面向 data parallelism 的。

Sia 还有一个基于 simple scaling 的 profiler，假设每个 job 的每个 DP worker 都可以在单张 GPU 上放得下 (不适用于 large-model)，在一张 GPU 上自底向上地增加 batch size 直到 OOM，从而获得单卡的 profiling data。在估计多卡性能时，Sia 忽略通信开销，直接线性地用 "GPU num * 单卡吞吐" 的方法来估计多卡吞吐，并进行 runtime 地修正。当 GPU type A 的 N 卡吞吐已经被 runtime 修正后，直接基于单卡吞吐用简单比例的形式来估计 GPU type B 的 N 卡吞吐：thr_N_B = thr_N_A * (thr_1_B / thr_1_A)。这其实是假设了模型在不同 GPU type 上进行 GPU num scaling 的时候的曲线是一致的，只能作为一种粗粒度的估计。

