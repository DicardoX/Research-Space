# (FlexFlow) Beyond Data and Model Parallelism for Deep Neural Networks

FlexFlow 在 OptCNN 的基础上，提出**包括 Sample（layer 的数据并行）、Operator（不同 OP 如何并行）、Attribute（样本高/宽等不同属性如何划分） 和 Parameter（channel 等模型参数如何在设备间分布） 在内的 SOAP 并行策略搜索空间**，一个**基于 profile 和理论计算的，对特定策略进行性能预测的增量执行模拟器**，以及**一个面向最优并行策略的基于引导随机和 MCMC 采样的增量搜索算法**。

###### Part 1. 并行相关概念的定义和任务图的构造

**关于并行维度、配置、策略的定义与 OptCNN 类似**，但**从面向 CNN 拓展到面向全体 DNN**，**考虑 OP 维度并行**体现在，**并行配置中考虑了不同 task 位于哪个设备**。FlexFlow 在 OptCNN 之外还**假设设备以 FIFO 处理任务**（为 OP 维度并行准备）。

与 OptCNN 不同的是，FlexFlow 在设计图搜索策略时没有局限在 OP 级别，而是**以 task 为粒度构造了更为细致的任务图（Task Graph）**。任务图**对从 OP 中产生的独立任务间的依赖进行建模**。**将硬件连接建模为通信设备**，**仅执行通信任务（如数据传输）**；**计算设备执行普通的计算任务**。由于所有设备独立，可进行**任务重叠**。任务图中，**节点是任务（计算或通信任务）**，**边是依赖关系**。注意，**边仅是顺序约束，并不代表数据流**（数据流作为通信任务被包括在任务图中）。任务图的构造有如下规则：

- 规则一：对于**配置 c 的 OP**，**划分为 |c| 个任务作为任务图内的节点**；
- 规则二：对于**每个 tensor (op_i, op_j)**，分别**计算 op_i 和 op_j 各自子任务的 sub-tensors**。若 **t_i 和 t_j 有共享 tensor 且两个任务在相同设备上**，则在任务图中**添加边 (t_i, t_j)**；若**位于不同设备**，则**先添加通信任务 t_c**，**再添加 (t_i, t_c) 和 (t_c, t_j) 两条边**。t_c 被分配在 t_i 和 t_j 所在设备中间的那个通信设备上。

------

###### Part 2. Cost 模型和任务图时间性能评估

与 OptCNN 类似，**对特定配置下 OP （task）的执行时间（t_c、t_x、t_s）进行建模（profile & 理论计算）**，并将**预测信息整合到任务图**中，作为相应任务的执行时间信息。相较于使用 **Full Simulation** 进行系统性能的模拟评估， FlexFlow 使用 **Delta Simulation 增量模拟算法**。具体来说，FlexFlow **从原来的任务图开始，仅重新模拟执行时间流改变的那部分任务**。

基于上述 SOAP 任务图的构建以及增量模拟方法，FlexFlow 针对不同的策略 S，给出预测的时间性能。

----------

###### Part 3. 面向最优并行策略的基于引导随机和 MCMC 采样的增量搜索算法

对于最优并行策略的搜索算法，FlexFlow 采用**基于引导随机和 MCMC 采用的增量搜索**。**MCMC 采样维护现有策略 S，随机提出（见下策略提出）一个新策略 S'，并根据 cost 模型（即 simulator 模拟得到的预测性能）来计算被采用的可能性**。MCMC 表现为**贪心搜索算法**，倾向于更低的 cost，但也可以**跳出局部最优解**。FlexFlow 的搜索算法**随机选择原有策略 S 的一个 OP**，**随机重新生成一个配置**。利用上述**策略提出（proposal）**的方法，FlexFlow 将已有策略（如专家定制的策略，数据并行策略等）和一个随机生成的策略同时**作为搜索的初始策略**，**开始搜索（迭代调用 proposal 和 MCMC 采样）直到搜索停止**。

-----

###### Part 4. 总结

整篇 paper 最 fancy 的地方在于**定义了面向全体 DNN 的 SOAP 多维并行策略搜索空间**，**以 task 为粒度的任务图的构建**，以及**基于引导随机和 MCMC 采样的增量搜索算法**。
