# Bagpipe: Accelerating Deep Recommendation Model Training

Bagpipe 是一个面向 Deep Learning Recommendation Model (DLRM) offline training (即基于已有数据集训练)，将 caching 和 prefetching 结合起来解决 embedding access 存在严重不均衡现象 (90% accesses 来自 0.1% 的embeddings，且随时间变化) 的训练系统。基于 offline training 中未来 batches 可预测的事实，Bagpipe 参考 perfect-caching 算法设计了一个 lookahead 算法 (Oracle Cacher)，在 x - L 个 iteration 前决策在第 x 个 iteration 时哪些 embeddings 应该被 evict，哪些 embeddings 应该被 prefetched。为了减少 model sync 并保证一致性，Bagpipe 实现了一个 Logically Replicate Physically Partitioned (LRPP) Cache，从 oracle cacher 角度来看，所有 cached embeddings 是 replicate 到每个 trainer 的；从 trainer 角度来看，一些仅一个 trainer 需要用到的 embedding 只由一个 trainer 进行 cache 和 update，并进一步采用了 Critical Path Analysis (CPA) 来仅更新部分 embeddings。进一步地，Bagpipe 采用了 delayed sync，仅立即更新下一个 iteration 需要使用且位于 critical path 上的 embeddings，其他 embeddings 的更新则和下一个 iteration 的 forward computation 进行 overlap。