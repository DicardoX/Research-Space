# FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU

FlexGen 是一个支持在单 GPU 有限 GPU memory 上进行 LLM inference 的 throughput-oriented 的系统，主要包括面向 GPU-CPU-disk 三级存储架构的 offloading 策略设计和 compression (Group-wise Quantization + Sparse Attention，不加介绍) 两个部分。对于 Offloading 策略，FlexGen 定义了两个维度：compute schedule 和 tensor placement。

对于 compute schedule，FlexGen 首先将 memory 首先的 inference 定义为一个图遍历问题 (包含多个 batches 和多个 tokens 两个维度，且每个 token 需要遍历多个 layers)。相较于传统的横向 batch-by-batch (KV cache 和 activations 可以在一个 batch 结束后释放，但反复的 layer 加载/卸载会带来很大 IO)，FlexGen 提出以 zig-zag block schedule 的方式进行多个 batches (一个 block 内，为了防止二级存储被占满，若占满则超出部分横向遍历) 相同 token idx 的纵向遍历，这样能使得一次 layer 加载可以被多个 batches 使用 (但似乎等价于 fuse 为一个更大的 global batch？或许可以将 seq_len 相似的 group 到一起来减少 padding 从而提高 compute efficiency)。

对于 tensor placement，FlexGen 以 layer 粒度 (i.e., x% 的 weights 放在 GPU/CPU/disk，overhead 更低) 对 weights 进行划分，以 tensor 粒度 (更灵活) 对 activations 和 KV cache 进行划分。此外，FlexGen 将 IO-bounded 的 attention 计算 offload 到 CPU 计算。

对于上述两个维度产生的超参 (GPU batch size, block size，各存储的划分百分比)，FlexGen 枚举前两者，并用 LP 来求解其余变量。FlexGen 对 inference latency 进行建模 (考虑计算和 IO)，通过 runtime profiling 来采样拟合硬件参数 (e.g., disk-to-cpu-bandwidth)。此外，FlexGen 仅支持简单的 multi-GPUs pipeline parallelism (L 个 layers 均等切分)。

