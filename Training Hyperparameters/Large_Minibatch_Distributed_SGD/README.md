# Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour

Facebook 提出的分布式训练下 Distributed SGD 过程在应用大的 minibatch 的同时，保证训练准确性的方法。

建立**学习率与 batch size 的线性函数关系**，每个 GPU/worker 的 bs 确定，通过调整 GPU 数目来改变整体的 minibatch size。由于**将求梯度时的 average 操作限制在 per-worker 级别**，**可以集成到 ring allreduce 等仅支持加法操作的梯度聚合方法上**。

在**通信方面**，本文**针对梯度聚合提出了三步 allreduce 操作**，依次是：

- 相同 server 上 GPUs 的 buffers 先加（为了接近线性 scale，由于不同 layers 之间的梯度没有依赖关系，可以将相同 server 上 workers 的梯度聚合和 bp 并行开展）
- 不同 servers 上的 result buffers 再加，
- 结果广播给每个 GPU。

**servers 间的通信算法**有两类：

- 算法一：**recursive halving and doubling 算法**。包括 log 级别的通信步数，相对更快。可以视为经典的二叉树形算法。
- 算法二：**bucket (ring) 算法**。包括一次方级别的通信步数，相对更慢。

整篇 paper 最 fancy 的地方在于建立了**学习率与 batch size 的线性函数关系**，可以通过 **profile 小的 minibatch，来评估大的 minibatch**，且**介绍了两类 allreduce 算法**。
