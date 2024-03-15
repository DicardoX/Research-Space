# SiloD: A Co-design of Caching and Scheduling for Deep Learning Clusters

SiloD 是一个 co-design 集群调度和任务 dataset caching (使用 local disks 来 cache 而不是全部通过 remote IO) 的调度框架，基于 training 数据访问的特性 (重复，可预测)，基于 uniform caching (LRU) 设计了一个考虑 cache 和 remote IO 的性能 estimator，并同时与传统的 computation-bound 的 estimator 一起使用，当发现为 IO-bound 时采用前者进行调度。在该 estimator 内，SiloD 提出了任务的 cache efficiency (remote IO / dataset size)，进而进行 estimator 的 analytical 构造。基于该 estimator，SiloD 将不同的调度策略 (multi-resource SJF, Gavel, greedy) 进行了适配，以提高了调度决策在 IO-bound 时的质量。SiloD 的 distributed cache 是基于 Alluxio 实现的，并在每个 host 上使用FUSE (Filesystem in USErspace) clients 进行 cache 的管理。

