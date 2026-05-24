# ORCA: A Distributed Serving System for Transformer-Based Generative Models

ORCA 是一个面向 LLM multi-iterations inference 的系统，insight 是 batch 内部分 request 可能会先完成 (output sequence 长度不同，每一个 iteration 仅生成一个 token)，而 request-level batching 会带来 new requests 的长时间等待。因此，ORCA 以 iteration 为粒度 (而非 request) 进行 request scheduling 和 batching。由于 attention layer 需要 batch 内所有 requests 的长度相同，因此 ORCA 选择不去对 attention operation (没有 model params，不会影响 model efficiency) 进行 request batching，而对其他 layers 仍进行 batching。
