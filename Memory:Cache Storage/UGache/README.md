# UGache: A Unified GPU Cache for Embedding-based Deep Learning

UGache 是一个面向 embedding 存储 (e.g., DLRM online training, GNN inference) 的 unified multi-GPU cache 系统，支持 non-uniform 和 switch-based 等多种 GPU topology 下的 unified embedding cache. 

相较于 replication-based (每个 GPU 独立地 cache 热门 embeddings，存在大量重复) 和 partition-based (尽可能多且均衡地在多 GPU 上划分并 cache 热门 embeddings，local hit rate 差) caching policy，UGcache 通过定义个 hotness metric (由用户指定，可以是第一个 epoch 的 access frequency，由于 embedding access 的 stable 和 predictable 特性)，再加上可用 GPU memory 以及带宽拓扑，以一个 MILP 问题的 formulation 通过求解器 (gurobi) 来求解 embedding cache 的 storage 和 access arragement (以最小化 max extraction time，并在 local hit rate 和 global hit rate 之间找到平衡)。

此外，UGache 通过一个 Factorized Extraction 机制来静态地指定 extract embedding entries 时不同 GPU topology 下的 GPU SMs 划分 (简单地按带宽成比例划分)，且将 embedding keys 按照 source location 进行 group，要求每个 GPU 优先 extract non-local group 以限制相同 source 的并行 extraction (保证 < N-1？不太懂哪里限制了，应该只是整体限制了不会出现太多的高并行度的 extraction，和前面的 storage arragement 也有关)，再以 local group extraction 填充不一样长的 extraction time (non-local group 读的越多，local group 可能会越少)。
