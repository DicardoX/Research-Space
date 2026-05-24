# Hydro: Surrogate-Based Hyperparameter Tuning Service in Datacenters

Hydro 是一个 surrogate-based 的超参搜索服务，能够在 job-level (Hydro Tuner) 和 cluster-level (Hydro Coordinator) 两个层面优化 tuning workloads，从而在提高 tuning efficiency 的同时提高异构集群利用率。Hyperparameter tuning 有两个特点：Trial throughput insensitivity (对部分 trials 性能下降不敏感，受制于 straggler) 和 Diminishing resource requirements (并行度和资源利用率随着越来越多 trials 的完成而下降)。

Hydro Tuner (user interface) 通过自动生成基于 Hyperparameter Transfer Theory 的小模型 (surrogate model，而非直接在大模型上，定义为 parametrization) 进行超参搜索，并 fuse 一次 tuning 内多组 trials 的 operators 来提高利用率 (inter-trial fusion，基于 JAX vmap)。Layer-wise intra-trial fusion 也被用来在特定情况下提高 throughput (由于 compiling overhead 未广泛应用)。此外，tuner 还使用 adaptive fusion (根据资源量决定 fusion count) 和 eager transfer (50% 的 trials 完成后即用当前最好的超参开始训练，若后续搜到更好的则重新开始训练)。

Hydro Coordinator (cluster inferface) 利用 pipeline 产生的 bubbles 来进行 trials 的 interleaving 以提高资源利用率，并参考 Gavel 进行 heterogeneity-aware 的资源调度。当部分 trials 完成并产生空闲资源时，coordinator 会进行空闲资源的弹性调度以充分利用分配的资源。

