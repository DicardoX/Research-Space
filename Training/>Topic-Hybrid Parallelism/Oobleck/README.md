# Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates

Oobleck 是一个面向 hybrid parallelism 且支持 fault tolerance 的训练框架，其核心 design 是 pipeline template，即在 model training 之前预先建立多个模版，每个模版对应不同的 pipeline nodes 数目，pipeline stages 数目和 layers 到 stages 的映射关系 (包括 intra-stage tensor parallelism)。Oobleck 的核心 idea 是在训练过程中，当一部分 pipelines (之间进行 data parallelism) 中的某些 nodes 出现 failure 时，可以从其他 pipelines 中尚未 failure 的 replicas 中之间拷贝 model states，而不需要 checkpointing (类似于构造了 data parallelism replicas 之间的 inherent redundancy)。需要注意的是，不同 pipelines 的 #nodes 和 parallelism plans 可能不同 (e.g., pipeline A 有 3 个 stages，pipeline B 有 4 个 stages)，所以这个步骤是以 layer 的粒度去实现的，包括不同 data parallelized pipelines 之间的 gradient sync。

Pipeline templates 的枚举和建立是通过一个 divide and conquer 算法实现的 (这过程中需要使用不同组合的 iteration time，无疑需要大量的 profiling，但论文中未提及)。在进行 pipeline reinstantiation 时，可以递进地分为三个阶段：(1) 若删掉 failed node 之后仍存在对应的 pipeline template，则直接按该 template 重启；(2) 若删掉之后小于所有 pipeline template 的最小 #nodes，则从其他 pipelines 中借用 nodes 并同时重启 failed pipeline 和 borrowed pipelines；(3) 若所有 pipelines 的 #nodes 都很少，则进行 merging。然后，Oobleck 会基于不同 pipelines 的 estimated execution latency 尽可能均衡地进行 batch distribution。

