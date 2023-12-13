# Gemini: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints

Gemini 是一个 enable 快速 failure recovery 的分布式训练系统，通过将每台 machines 上的 checkpoints 存储到 local CPU memory (as L1 cache) + remote CPU memory within the same group (as L2 cache) + remote persistent storage (RPS, as memory) (而不是简单地存储到 RPS)，利用 GPU-to-CPU 的高带宽实现快速且频繁的 checkpoints 存储，以及高效的多级 checkpoints 读取 (类似于多级 cache 和 memory 的方式)。

Gemini 需要解决两个问题：(1) 如何最大化从 CPU memory 可以恢复 checkpoints 的概率 (而不需要 fetch from RPS)？(2) 如何减少 checkpoint traffic 和 training traffic (gradient sync + parameter fetching) 的干扰？对于前者，Gemini 主要使用了 group placement strategy，将 N 个 machines 划分为 M 组，组内互相 broadcast 冗余的 checkpoints，并从理论上证明了最优性。对于后者，Gemini 将 checkpoint communication 与 computation overlap (基于 training traffic 与 computation 无法 overlap 的观察)，并将 checkpoint 进行细粒度 partition，以 pipeline GPU-to-GPU remote copy -> GPU-to-CPU copy (on remote machines)，并减少额外预留的 GPU memory (128 MB in implememtation)。
