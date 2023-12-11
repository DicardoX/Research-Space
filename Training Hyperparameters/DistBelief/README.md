# (DistBelief) Large Scale Distributed Deep Networks

DistBelief 是 Google 在 2012 年提出的一个**支持多机分布式训练的软件框架**，**第一次对大模型提出了模型并行（Model Parallelism）的方法**（包括不同 layers 分布在不同 machines 上，以及相同 layer 中的不同子 tensors 分布在不同 machines 上），**和数据并行结合**，面向**在线和批处理场景**提出了**两类算法**。

注意，只有包含跨越 machines 的边的 nodes 才需要在 machines 间传输状态。**DistBelief 框架也能够支持数据并行**，且**揭示了异步 SGD（之前很少用在非凸问题）在分布式训练中表现良好**，特别是和 Adagrad 自适应学习率方法结合时。

DistBelief 主要由两个算法构成。算法一是 **Downpour SGD**，一个**异步 SGD** 过程，能**自适应学习率**，**支持大规模模型副本**；算法二是 **Sandblaster L-BFGS**，**L-BFGS 的分布式实现**，**同时使用数据和模型并行**。

- **Downpour SGD 算法**：**在线场景**。SGD的传统公式本质上是顺序的，因此不适用非常大的数据集。Downpour SGD 就是**异步 parameter server (PS) 架构的模型并行版本**。不同的 workers（都保存一份独立的 replica，用独立的数据集进行 fp 和 bp）之间和不同的 parameter server shards 之间都是异步的。由于 workers 被划分到不同 machines 上，因此**每个 machine 只需跟一部分 ps shards 通信**。注意，**machine 发送和接收时需要进行同步**，以 replica 为单位，**尽量避免（但没有保证）算力差别导致的异步累积**，即 ps shards 之间的参数更新代数不同，这会带来更多的随机性。**放宽一致性在非凸问题中并无理论依据，但在实际中非常有效**。对于 ps shards，**异步体现在更新更快的 shards 先把这部分更新后的参数返回给 worker 中特定的 machine**。**Adagrad 自适应学习率策略直接在 ps shard 中**，用梯度来算学习率，易实现。该策略能够扩展 model replicas 能实际使用的个数，且与 “少数几个 replicas warmstarting，再逐步加入其他 replicas” 策略结合使用时，能够很好地解决 Downpour SGD 训练时的稳定性问题。

- **Sandblaster L-BFGS 算法**：**批处理场景**。**Sandblaster 框架**下，**优化算法（如 L-BFGS）在协调进程中**，该进程**并不直接访问模型参数**，而是**向 ps shards 发送一系列子操作**（点积，放缩，考虑系数的加，乘等），依次进行一个个 batch 处理的优化，计算结果保存在 shard 本地。这样做可以**避免需要把所有参数和梯度都汇聚到一个中心 server 上**，这也是模型并行和数据并行 replica machines 和 ps shards  “多对多” 的优势。

    为了缓解短板效应，提出了一个**负载均衡策略**：协调器**给每个 replica 分配一个很小比例的工作（相较于 1/N batch）**，并**给那些完成比较快的 replicas 分配新的更多工作**。对于 batch 最后的工作，协调器会让多个 replicas 同时运算，并采用完成最快的那个。

    上述策略意味着，**Sandblaster L-BFGS** 无需**像 Downpour SGD 那样将 dataset 划分为多个独立的 subset**，而是**以一个个 batch 的形式，在协调器的指挥下，供多个 replicas 进行处理**。

整篇 paper 最 fancy 的地方在于**第一个提出了模型并行的设计**，并**与数据并行相结合**（优势在于避免把所有参数和梯度都汇聚到一个中心 server 上），且分别**对在线和批处理两类情景提出了两类算法**。
