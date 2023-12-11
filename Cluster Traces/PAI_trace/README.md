# MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters

本文研究了 Alibaba 工业 MLaaS 集群 (超过 6,000 GPUs) 下两个月的负载 trace，已开源。

集群调度的挑战包括：
1) 低 GPU 利用率
2) 长排队时间
3) 资源需求严格，难以调度的任务的存在
4) 异构设备间的负载不均衡
5) CPU 潜在的算力瓶颈
