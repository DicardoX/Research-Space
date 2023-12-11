# Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads

- 研究 **locality-aware 的调度如何影响性能和利用率**，包括：(1) Locality 约束带来的等待如何影响排队延迟 (放宽 locality 约束可以减少排队延迟，尤其是占用很多 GPUs 的 jobs)；(2) Locality-aware 调度如何影响分布式训练 jobs 的 GPU 利用率 (跨 servers job 的同步开销 + 相同 server 上混布多 jobs 干扰影响 (非相同 GPU 混布) -> GPU 低利用率)。
    - **Fair-share Delay**: 由于 fairness，即 VC 用完了分配的 quota，可通过抢占消除；
    - **Fragmentation Delay**: 资源碎片化导致无法找到高 locality 的足够资源，不好消除。
        - <font color=blue> 对于大任务，fragmentation delay 占比更大，这也印证了放松 locality 来缓解排队延迟的可行性。</font>
        - 乱序调度 (e.g., 某需求量大的 job 排队时，先让小 job 暂时占用) 对资源密集型 jobs 的排队时间影响不大。
        - 放松 locality 需求来缓解排队延迟对于分布式训练是合理的。<font color=blue> 这对我们的工作也是一个可以引用的观点。</font>
- **调度器设计策略的建议**：(1) locality 差同时影响性能和硬件利用率，用排队延迟来换取遵守 locality 约束，但也可以适当 relax locality 来降低延迟 (等待一段时间后再放松)，对于长时任务可以适当严格 locality 要求，或者进行 runtime migration 来改善 locality；(2) 使用任务迁移等技术让多 GPU jobs 尽量独占 servers (缓解小任务之间的干扰，提高大任务 intra-job locality，优先保证后者)；(3) 许多错误应该被提早发现，使用少量资源来预验证多 GPU jobs 程序和配置的正确性
- **Failure Handle**. 
