# Egeria: Efficient DNN Training with Knowledge-Guided Layer Freezing

Egeria 是一个基于 knowledge distillation 进行 layer freezing 的 DNN 训练框架，周期性地对模型参数量化并保存为一个 reference model (分为 (bootstrapping (初始化) and knowledge-guided (周期性更新) 两个阶段) 与原模型进行中间结果对比，进而决定哪些 layers 可以被提前 freeze (观察是训练时前面的 layers 要更早地趋于收敛)。具体来说，Egeria 通过少量 CPU 资源对 reference model 进行前向 (non-blocking + async，可以被 overlapped) 并获取 intermediate activations (相较于 gradient 等更能反映模型特征)，与原模型各 layer 的 intermediate activations 进行 SP loss 的计算，若一段时间内趋于稳定则冻结该 layer。注意，Egeria 仅支持同时冻结前 k 个 layers (不可以间隔冻结)，因为 frozened layer 仅进行前向，而后向也是需要传播梯度信息的。



