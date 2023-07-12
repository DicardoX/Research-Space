# Jax, Jaxprs, HLO IR, Graph Optimization in XLA Compilation and Runtime

> 作者：Dicardo  dicardo@sjtu.edu.cn
>
> 摘要：本文介绍了深度学习框架 JAX 从 language level 到 optimized compilation level，再到 XLA runtime level 的执行工作流，并讨论了 GPU 硬件对 XLA runtime graph optimization 的影响。

### 1. Terminology

#### 1.1 JAX 编程框架

对于 JAX，官方文档 [1] 中给出了如下定义："JAX is [Autograd](https://github.com/hips/autograd) and [XLA](https://www.tensorflow.org/xla), brought together for high-performance numerical computing"。其中，Autograd 为深度学习中的梯度求导过程提供了自动微分功能，而 XLA backend 则作为 JAX 程序的编译和运行的优化和执行引擎，并支持 GPU 加速。

因此，一言蔽之，**JAX = Numpy (数学运算) + Autograd (自动微分) + XLA (编译优化和执行引擎)**.

需要注意的是，与 TensorFlow 框架下直接在代码中定义静态图，再启用 `session.run()` 进入迭代运算不同，JAX 与 Pytorch 类似使用**动态图**的形式，将运算与搭建过程同时进行 (这也符合 NumPy 数学运算的实现逻辑)。虽然上述特性使得 JAX 编程框架拥有与 Pytorch 近似的用户亲和性和灵活性，但也存在一个问题，即**动态图无法直接利用 ML compiler JIT 编译时面向计算图的图优化** (e.g., 算子融合)，从而错失代码高效编译和执行的机会。

#### 1.2 Jaxprs

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

#### 1.3 HLO IR in XLA



#### 1.4 XLA Compilation



#### 1.5 XLA Runtime



---------



### 2. Workflow



---------



### 3. References

[1] Jax Official Documentation. https://jax.readthedocs.io/en/latest/index.html

[2] 一文带你掌握深度学习框架 Jax. https://www.modb.pro/db/161900

[3] Understanding Jaxprs. https://jax.readthedocs.io/en/latest/jaxpr.html

[4] HLO-XLA. https://blog.csdn.net/weixin_45387966/article/details/121994883



[x] XLA Official Documentation. https://www.tensorflow.org/xla
