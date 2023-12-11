# Discussion about The Stability of the profiling data with Alpa



-------

## 1. Issues

- When I use Wide-ResNet as the benchmark with relatively low workload (e.g., param num = 500M on 4 V100 GPUs on the single host), why Alpa prefers to perform pipeline parallelism rather than data parallelism (in Bert/MoE, data parallelism is much more frequent than pipeline in low workload case)? I have set the num of micro batches to 2 for all experiments, does this affect the result? 

    - B.T.W., according to your objective function (minimize e2e pipeline latency), I think less micro batch num can lead to less pipeline stages?

    - **Answer**: **A too small microbatches may make the Alpa's formulation not correct**, which assumes there is always enough microbatches to fill in the pipeline. So it also does not consider to optimize the pp stages in that case.

        Besides, for the Wide-ResNet, our default benchmark script turns off the kernel tuning in the final compilation, which may hurts the performance(ref: https://github.com/alpa-projects/alpa/blob/efe417d86327b87fed4c1f9c7df92f53a525c62d/benchmark/alpa/benchmark_one_case.py#L69), but I'm not sure if it will make the result random. To turn it on, please set it back to 4.

- I have repeated the same configurations on the same hardwares, but got some completely different result. Is this normal? Or maybe because some hardware reasons (e.g., interfered by other co-located jobs in the same host). 

    - **Answer**: We haven't met the unstable case, and sorry we may be unable to give you more helpful information. If I were you, I'd try to 1) manually set the configuration to the final result, then run multiple rounds and check what happens; 2) print the profiled output matrix and check whether it differs a lot in two runs. Besides, the multi-tenancy you mentioned might be a reason. During the profile time, Alpa does not guarantee that all devices are fully owned by itself. Instead, it creates a ProfileWorker and later destroys it. Then creates another for the next profiling.

- So as I set the micro batch num to 2 for all configurations (e.g., 4 V100 GPUs, 8 V100 GPUs), the optimized result (parallel method and the e2e iteration time) I got is actually unreliable? Does setting micro batch num to 2 lead to less pipeline parallelism? If the optimized result is to perform DP or MP only, I wonder setting this to 2 can still affect the correctness of Alpa? 
- What I should do is to make sure that the micro batch num is no less than the stage num (in order to fill in the pipeline)?



-------

## 2. Opportunities

