# Tabi: An Efficient Multi-Level Inference System for Large Language Models

Tabi 是一个利用 multi-level model inference 来降低 LLM 推理延迟的推理框架，对象是 descriminative model (而非生成式) 和 single data inference，其观察是由于增加模型参数的边际递减效应，一个小模型可以在大多数 queries 上取得和大模型一样的 accuracy。Tabi 通过 offline profiling 来获得多个 model candidates 的相关数据 (accuracy，latency，中间结果)，并组合成一系列 candidates (combinations with multi model levels)，并在 online candiate selection 时选择那个能满足 task accuracy 和 latency 需求的 latency 最小的那个 candidate。

对于一个 query，Tabi 需要决策是否直接返回小模型的结果作为输出，还是 re-route 到下一个 model level (一个更大的模型)。Tabi 在 softmax (其概率天然表示了每个 token 的 confidence score) 之上使用了 temperature scaling 来校正 (calibrate) confidence score (选择某个 token 的概率)。基于该 calibrated confidence score，Tabi 定义了一个 scaled sigmoid 风格的概率函数，来表示将 query re-route 的概率。此外，Tabi 还会对 re-routed queries 进行 word pruning，基于 profiling 获得的 attention weights 计算该 query 的 importance vector (表征每个 token 与其他 tokens 的关联程度)，在加权均值二分后决定哪些 token 可以被 pruning (对于 tokenizers 必须该 token 下的所有 sub-tokens 都可以被 pruning)。此外，Tabi 还会对 rerouted-queries 进行加权的结果装配和输出，及将该 query 所有经过的 model levels 的输出进行加权和作为最终输出的 predictions，权重与每个 model level 的 accuracy 成比例。

