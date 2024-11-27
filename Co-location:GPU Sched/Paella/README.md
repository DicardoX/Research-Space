# Paella: Low-latency Model Serving with Software-defined GPU Scheduling

Paella 是一个 co-design compiler-library-scheduler 的模型推理框架，实现了 bypass GPU hardware scheduler 的 runtime 黑盒调度，以及各组件之间的低延迟通信。Paella 由三个部分组成：compiler, client (RPC server with communication protocol) 和 dispatcher。

Paella 通过一个 built-on TVM 的 compiler 来获取 kernels 的实时信息 (block size 等静态信息可以直接在 kernel lanuch 之前获取，SM allocation 等实时信息则由 hardware scheduler 决定，必须在 lanuch 之后获取)。基于上述 kernel 和 GPU occupancy 信息，Paella dispatcher 进行支持多个 scheduling objectives 的 software-defined 调度，通过一个 per-job 的 kernel waitlist 将每个 job 的 kernels 存储起来而非直接 submit 到 GPU 的 hardware scheduler (该步骤通过 warp CUDA functions 实现)，并使用 Boost coroutines 来低上下文切换开销地实现 cooperative multitasking (原因是 multi-jobs 场景下会存在 blocking calls 和 forced preemption)。当调用一个同步 CUDA 函数时，当前协程上下文会 yield；当异步调用时，当前协程继续执行。Client 通信则不再关注，主要就是通过 IPC socket 通知 client 任务完成再由 client 用 RPC 获取结果，不干扰 inference。

