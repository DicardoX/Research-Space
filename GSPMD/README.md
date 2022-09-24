# GSPMD: General and Scalable Parallelization for ML Computation Graphs

GSPMD 是一个自动化，compiler-based 的并行系统，允许 users 通过少量指示如何 distribute tensors 的标注，像单机那样编程。GSPMD 的 OP 划分表示简单有效，可表示不同或混合类别的并行，包括 DP，in-layer MP，spatial parallelism 和 weight update sharding，并通过一个 wrapper library 将 pipeline reduce 为一个 tensor/OP 划分问题

Contributions：

- GSPMD 为所有划分方案生成单个程序，即 Single Program Multiple Data (SPMD)，这对 scale 到更多划分很重要；

- GSPMD 支持非均衡划分的维度，允许任何 tensor 在任意 device meshes 上划分；

- GSPMD 被使纤维 XLA 编译器的扩展，后者是一个涵盖多架构和多硬件平台的统一抽象，这使得 GSPMD 是可复用的；

- GSPMD 支持嵌套模式的并行：在 OP level，不同类别的维度可以在正交的 devices 组间划分。GSPMD 使用一个递归方法来实现上述嵌套模式，在无需手写额外划分规则的同时最大化通用性。

这篇 paper 主要介绍了 tensor sharding propagation 和 partition 过程中的一些 tricks，以及 weak pipeline method。
