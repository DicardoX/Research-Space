# ZeRO: Memory Optimization Towards Training A Trillion Parameter Models

**ZeRO 优化器**提出了 **weight update sharding** 方法，可以**优化内存，消除 data parallelism 和 model parallelism 的内存冗余**；同时，ZeRO 可以在加快训练速度的同时，**增大可有效训练的模型 size，并使其与设备数目成比例增加**；此外，ZeRO 可以保持**较低的额外通信开销（为了换取内存优化）**，且指出 **ZeRO 对于通信延迟的影响较小**，且相较于通信量和通信带宽，延迟对训练速度的限制更小。

ZeRO 优化器由 ZeRO-DP 和 ZeRO-R 两类应用场景组成：

- **ZeRO-DP**：包括 P\_{os}（distribute 优化器状态）、P\_{os+g}（distribute 优化器状态和梯度）和 P\_{os+g+p}（distribute 优化器状态、梯度和参数）三类：
    - **P\_{os+g}** 方法中，**每个 worker 仅需要更新自己负责的那部分参数**（尽管均保存了全部参数），因此**仅使用 scatter-reduce 操作（ring allreduce 的前一半）来进行梯度同步**，**通信量为 M**；各自部分的参数更新完后，使用 **all-gather（ring allreduce 的后一半）进行参数同步**，**通信量为 M**；**总通信量为 2M**，由于仅更新负责那部分的参数，因此存储 1/Nd 的优化器状态和梯度是合理的。
    - **P\_{os+g+p}** 方法中，ZeRO **重构了 P_{os+g} 中参数 all-gather 的操作**，**在 fp 时需要在计算特定部分模型时将相应 worker 上的参数广播给全部 workers**，并在计算完该部分的 activations 后将其他 workers 上的这部分参数丢弃；**在 bp 时，ZeRO 同样进行类似的 all-gather 操作**，但**每个 worker 仅长期保留负责那部分参数的梯度**，用作后续的部分参数更新，其他部分的梯度要么丢弃（之前，第 N 层的梯度计算和 N+1 层的梯度无关，仅和过程中的部分偏导结果有关，因此可以 pipeline），要么直接不计算（后面）；**总通信量为 3M**，**由于大模型在单层计算上的时间开销较大，通过 pipeline 实现的参数实时广播的开销是可以被 overlap 的**。**可能存在的通信延迟是由包含计算和参数广播的 pipeline 造成的**（比如计算算完了，下一层参数还没全部广播完，造成 bubbles） 。 
        - **fp 时，all-gather 点对多实时广播参数，通信量为 M；bp 时进行相同的 all-gather，通信量为 M；利用 scatter-reduce 进行梯度同步，通信量为 M。由于在算出梯度之前已经进行了参数同步，因此之后不需要再进行，只需利用各自 scatter-reduce 获得的部分梯度，更新完负责的部分参数即可。**
        - **由于 worker 不保存完整的参数，因此后续不再需要进行一次 all-gather 以同步参数。**
- **ZeRO-R**：目的是**优化剩余内存**，包括 **activations 消耗的内存**，**临时 buffers** 和 **未使用的内存片段**。对于 activations 消耗的内存，ZeRO-R 通过 **activations 划分来识别并移除现有 model parallelism (MP) 方法里的 activation replication**；对于临时 buffers，ZeRO-R **适当地确定临时 buffers 的 size**；对于未使用的内存片段，ZeRO-R **主动管理不同生命周期的 tensors 内存，避免内存碎片化**。
