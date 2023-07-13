# Jax, Jaxprs, HLO IR, Graph Optimization in XLA Compilation and Runtime

> 作者：Dicardo  dicardo@sjtu.edu.cn
>
> 摘要：本文介绍了深度学习框架 JAX 从 language level 到 optimized compilation level，再到 XLA runtime level 的执行工作流，并讨论了 GPU 硬件对 XLA runtime graph optimization 的影响。

### 1. Terminology

#### 1.1 JAX 编程框架

对于 JAX，官方文档 [1] 中给出了如下定义："JAX is [Autograd](https://github.com/hips/autograd) and [XLA](https://www.tensorflow.org/xla), brought together for high-performance numerical computing"。其中，Autograd 为深度学习中的梯度求导过程提供了自动微分功能，而 XLA backend 则作为 JAX 程序的编译和运行的优化和执行引擎，并支持 GPU 加速。

因此，一言蔽之，**JAX = Numpy (数学运算) + Autograd (自动微分) + XLA (编译优化和执行引擎)**.

需要注意的是，与 TensorFlow 框架下直接在代码中定义静态图，再启用 `session.run()` 进入迭代运算不同，JAX 与 Pytorch 类似使用**动态图**的形式，将运算与搭建过程同时进行 (这也符合 NumPy 数学运算的实现逻辑)。虽然上述特性使得 JAX 编程框架拥有与 Pytorch 近似的用户亲和性和灵活性，但也存在一个问题，即**动态图无法直接利用 ML compiler JIT 编译时面向计算图的图优化** (e.g., 算子融合)，从而错失代码高效编译和执行的机会。

#### 1.2 Jaxprs IR

为了解决 1.1 节中 JAX 框架动态图设计带来的问题，JAX team 提出了一类**面向 JAX 框架下程序计算特性的抽象：Jaxpr 中间表示** (IR)。Jaxpr 提供了一项十分重要的功能：**追踪 (trace) JAX 程序的 API 调用链**。Jaxpr 的表达形式类似于汇编语言，如下所示：

```assembly
{
    lambda inputs ; params.
    let (b, c) = params
        (d, e) = b
        g = dot inputs d
        h = add g e
        i = tanh h
        (j, k) = c
        l = dot i j
        m = add l k
    in m
}
```

参考 [2] 中提到：

> Jaxpr 的每一条指令，都是对模型搭建 API 的 trace 结果，记录了其参数及其 shape，以及所用计算函数等。然后 Jaxpr 才会被转换成 HLO，交给 XLA Backend 做编译优化执行。
>
> 所以，整个 trace 的过程并没有涉及到真正的编译，计算等，而是纯纯粹粹的前端记录过程，可以认为是一种假执行过程。这一点非常重要，通过这种方式，Jax 能够从头到尾将整个计算逻辑全部 trace，使编译过程有全局角度，从而采取更加激进的优化策略。与此同时，由于只是做 trace，所以实现了在不真正消耗资源的情况下，获得了程序执行的全部静态信息。

一言蔽之，**Jaxpr 通过对 JAX 程序 (Python 函数) API 调用的 trace，将 JAX 计算流抽象为许多原子计算操作，使用 Python 解释器完成大部分代码端的解释工作 [3]，在不涉及编译的情况下提供了类似于静态图获得程序静态执行流的功能，并进一步被转化为 HLO IR (可以描述 JAX 程序的计算图)，作为 XLA 编译的输入**。

根据 JAX 官方文档中关于 Jaxprs 的介绍 [3]，一个 **Jaxpr 实例**表示具有一个或多个输入变量 (typed) 和一个或多个结果 (typed) 的函数，其结果仅依赖于输入变量。输入和输出变量具有类型，在 JAX 中表示为抽象值。

在 Jaxpr 的帮助下，JAX 既能够为 DL 开发者提供用户友好的 Pythonic 编程接口 (Pytorch 的长处)，也能够支持高效的编译优化和程序执行 (TensorFlow 的长处)。

#### 1.3 HLO IR in XLA

HLO (High-Level Optimizer) 是**作为 XLA compiler 输入的一种中间表示 (IR)**，负责将输入的计算图转换为一种中间表示形式，该表示形式可以进行各种优化操作 (e.g., 算子融合、常量折叠、循环展开、内存重用、并行化)。

参考 [4] 中对 HLO 的架构进行了如下总结：HLO IR 表示为**分层嵌套的结构**：HloModule, HloComputation和HloInstruction。

- HloModule 是 HLO IR 最顶层的表示，相当于整个程序。比如，我们可以用一个 HloModule 来表示一整个model。一个 HloModule 可以包含很多个 HloComputation。
- HloComputation 是 HLO IR 中间层的表示，相当于程序中的一个函数。一个 HloModule 只能有一个 entry_conputation，其他的 computation 是被 entry_computation 调用的。我们可以把 entry_computation 类比作main函数。每个 HloComputation 可以包含多个 HloInstruction.
- HloInstruction 是 HLO IR 最底层的表示，相当于程序中的一条指令。每个 HloComputation 只能有一个 root_instruction。root_instruction 的 output 就是该 computation 的 output。computation 的 input 用parameter 表示。HloInstruction 也可以调用 HloComputation。

一个 HLO fusion 的例子：在做 fusion 优化时，可以把多个要 fuse 的 instruction 添加到 fusion computation 中，然后在原来的 computation 中用一个 HloFusionInstruction 代替所有被 fuse 掉的 instruction，然后让这个 fusion instruction 去调用这个 fusion computation。这样，被 fuse 掉的 instruction 还会保留在 fusion computation 中，在 codegen 的时候才可以根据这些详细的信息去生成 code。

#### 1.4 XLA Compilation

XLA 官方文档 [5] 中给出如下介绍：**XLA 接受在 HLO 中定义的计算图（“计算”）并将其编译为适用于各种架构的机器指令**。XLA 采用模块化设计，可以轻松融入其他后端以[针对某些新颖的硬件架构](https://www.tensorflow.org/xla/developing_new_backend?hl=zh-cn)。XLA 提供了多种**与目标无关的优化和分析过程**（例如 [CSE](https://en.wikipedia.org/wiki/Common_subexpression_elimination)）、与目标无关的运算融合，以及用于为计算分配运行时内存的缓冲区分析。

完成与目标无关的步骤之后，XLA 会将 HLO 计算发送到后端。后端可以执行进一步的 HLO 级优化，而此时将**考虑目标特定的信息和需求而进行进一步的分析优化**。例如，XLA GPU 后端可以执行特别有利于 GPU 编程模型的运算融合，并确定如何将计算划分为计算流。在此阶段，后端还可能对某些运算或运算组合针对优化库调用执行模式匹配。

下一步是针对特定目标生成代码。XLA 所含的 CPU 和 GPU 后端使用 [LLVM](http://llvm.org/) 进行低级 IR、优化和代码生成。这些后端发出有效表示 XLA HLO 计算所需的 LLVM IR，然后调用 LLVM 以从此 LLVM IR 中发出原生代码。

#### 1.5 Operator Fusion in XLA

下面，我们参考论文 [6] 给出编译过程中 XLA 在 Operator Fusion 方面的详细设计。

##### 1.5.1 Background

传统的 ML 框架 (e.g., PyTorch, TensorFlow, MXNet) 一般将 DL operations 映射到 cuDNN/cuBLAS primitives 或预实现的 CUDA kernels，但这样无法保证 ML 程序的完整优化。随后，一些 DL 优化框架 (e.g., XLA, TensorRT, TVM, Tensor Comprehensions) 被设计以支持生成或使用 **workload-specific kernels**。例如，XLA 和 TensorRT **使用一些人工定义的规则来 fuse 一些简单的 operations**，而**复杂 operations (e.g., 卷积，矩阵乘) 依然依赖于 cuDNN/cuBLAS primitives**。另一方面，TVM / Tensor Comprehensions 的 codegen 更为灵活，可以使用一些学习算法 (e.g., GBM, genetic) 自动地 tune fused kernels。

##### 1.5.2 XLA 计算图优化

XLA 的计算图优化发生在 TF 或 JAX 的 traced computational graph 被输入到 XLA compiler 时。一些优化 passes 会被复用，例如 Dead Code Elimination (DCE) 和 Common Subexpression Elimination (CSE)。

Optimization passes 被组织为 **Pass Pipelines**，包括如下 pass：

- **SPMD partitioner**: 将 tensors 划分以在多设备上并行执行。
- Optimization: 包括 canonicalization, expansion, and simplification 等 passes。
- Simplification: 对特定操作执行简化，例如内联和常量传播。
- **Collective optimizations**: 优化 SPMD 多设备划分产生的 collective 操作 (e.g., reduce, gather)。
- Conv canonicalization.
- Layout assignment: 预先分配一些操作数的 layout，以满足 layout 约束和库调用的结果 (e.g., cuDNN, cuBLAS)。
- **Post layout assignment**: 在 layout assignment 后执行特定于目标的 HLO 优化过程，例如，优化 cuBLAS 的 padding 和选择 GEMM 或 Conv 算法。
- **Fusion**: 进行多种纵向 operation fusion，包括简单的 instruction fusion, fusion merger 和 multi-output fusion。
- **Horizontal fusion**: 进行横向 operation fusion，包括横向 loop fusion 和 input fusion。
- **Post fusion optimization**: 将小的独立 collective operations 组合为更大的 operations。
- GPU IR emit prepare: 对给定的 HLO 模块进行清洁 sanitize，以便其被 IR Emitter 接受。

##### 1.5.3 XLA Fusion Strategies

<img src="./figures/截屏2023-07-13 17.15.11.png" alt="avatar" style="zoom:50%;" />

- **Instruction Fusion**: 简单的纵向 fusion，producer instructions 被 fuse 进其 consumers。XLA 在此步骤中执行反向后序遍历，以确定是否应融合两个相关 operations。注意，XLA 会检查 fused kernel 对于 GPU 是否过大 (即**检查 GPU 硬件信息**)，并保证不会超过 threads per block, shared memory per block, and threads per SM 等 **GPU 硬件限制**。因此，**XLA 的图优化与 GPU 硬件相关**。
- **Fusion Merger**: 试图合并 fusion instructions 以减少内存带宽需求和内核启动开销。
- **Multi-Output Fusion**: GPU backend 的 sibling instruction 和 producer-comsumer instruction 的多输出融合也旨在降低内存带宽需求。
- **Horizontal Fusion**: 以减少 kernel lanuch 开销，同时增加 GPU 的 kernel lanuch 维度。例如，考虑具有单独输入 shape 的乘法运算和加法运算，但它们的输出被一个公共运算所使用。

#### 1.6 XLA Runtime

论文 [6] 中对 XLA kernel scheduling 和 CUDA streams 有如下讨论：

> At compile time, XLA’s IrEmitter also generates KernelThunks which contain necessary arguments for launching kernels. At run- time, GpuExecutable launches the kernel using the KernelThunk which specifies the buffer addresses of the data needed for the kernel launch. An initial finding is that the function BFSLaunchOrder() computes a topological launch order that is close to a breadth-first order. This enables the possibility of launching kernels concurrently in different CUDA streams. The function CanRunConcurrently() returns whether the two HLOs can run concurrently, however in practice we have not seen multiple streams utilized by XLA.

一言蔽之，**XLA runtime 使用 `BFSLanuchOrder()` 来以广度优先搜索 BFS 的方式计算一个拓扑上的 lancuh 顺序，从而支持在不同 CUDA streams 上并行地 lanuch kernels，但实际上 multi-streams 在 XLA 的使用不多**。

---------



### 2. References

[1] Jax Official Documentation. https://jax.readthedocs.io/en/latest/index.html

[2] 一文带你掌握深度学习框架 Jax. https://www.modb.pro/db/161900

[3] Understanding Jaxprs. https://jax.readthedocs.io/en/latest/jaxpr.html

[4] XLA笔记(1) -- HLO IR Introduction. https://zhuanlan.zhihu.com/p/396309457

[5] XLA Official Documentation. https://www.tensorflow.org/xla

[6] Operator Fusion in XLA: Analysis and Evaluation. https://arxiv.org/abs/2301.13062
