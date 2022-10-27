### Note for the Source Code of Pollux (AdaptDL)

[Link for source code](https://github.com/petuum/adaptdl)

> NOTE: The following file path description is based on the root path of `./adaptdl`, which is the path of pollux source code and the ray part.

Components of adaptDL:

- **adaptdl:** A library for adaptive batch sizes that can efficiently scale distributed training to many nodes.
    - Path: `./adaptdl/adaptdl`
    - Goodput function (`./adaptdl/adaptdl/goodput.py`)

- **adaptdl-sched:** A cluster scheduler on Kubernetes optimized for distributed deep learning training.
    - Path: `./sched/adaptdl_sched`

---------

Several Problems to be solved:

- How Pollux Agent (job-level) retrieve information?
    - 怎么实现：**PolluxAgent 收集 bs 和 $T_{iter}$ 等信息**，基于信息**获取 efficiency 和拟合 throughput 的函数**，进而**获取每个 job 的 goodput 函数**
- How Pollux Agent: 通过**最大化 job 的 goodput 来动态调整 bs 和 lr**，以更好地利用资源
- How PolluxSched (cluster-level): **基于 jobs 的 goodput 动态重分配资源**，通过**最大化 fitness 函数（由相较于 fair allocation 的 speedup 构造）来获取理论最优的分配**。
- How PolluxSched: **考虑多个集群层面的目标**，包括 fairness，goodput，reallocataion penalty，interference slowdown 等。
- How Pollux interact with hardware resources (maybe K8S pods)？

-------

#### 1. `./adaptdl/adaptdl/goodput.py`

- Parameters:

    ```python
    # Parameters for a performance model which predicts the per-step time of
    # distributed SGD using all-reduce. At a high level, models compute time and
    # network time separately, and combines them with some degree of overlap.
    # Compute time is modeled as a linear function of the local batch size.
    # Network time is modeled using different parameters depending on if the job
    # is inter-node (there exists a pair of replicas on different nodes), or
    # intra-node (all replicas are on the same node). For both cases, network time
    # is modeled as a constant term plus a retrogression term which increases
    # linearly with the total number of replicas.
    PerfParams = collections.namedtuple("PerfParams", [
        # T_compute ~ alpha_c + beta_c * local_bsz +
        #             (alpha_a + beta_a * local_bsz) * accumulation_steps
        "alpha_c",  # Constant term of compute time
        "beta_c",   # Multiplicative factor of compute time
        # If inter-node: T_network ~ alpha_n + beta_n * replicas
        "alpha_n",  # Constant term of inter-node network time
        "beta_n",   # Retrogression factor of inter-node network time
        # If intra-node: T_network ~ alpha_r + beta_r * replicas
        "alpha_r",  # Constant term of intra-node network time
        "beta_r",   # Retrogression factor of intra-node network time
        # T_step ~ (T_compute ^ gamma + T_network ^ gamma) ^ (1 / gamma)
        # Essentially is a p-norm where p = gamma. When p ~ 1 then
        # T_step ~ T_compute + T_network, indicating no overlap between compute
        # and network. When p -> infinity then T_step = max(T_compute, T_network),
        # indicating perfect overlap. We limit gamma to [1, 10] since 10 is close
        # enough to approximate the max function for our purposes.
        "gamma",    # Models the degree of overlap between compute and network
    ])
    
    GradParams = collections.namedtuple("GradParams", ["sqr", "var"])
    ```

- `class GoodputFunction(object):`

    - `def evaluate(self, num_nodes, num_replicas, atomic_bsz, accum_steps):`

        - Calculate overall batch size;
        - Call `self.throughput()` and `self.efficiency()`, return the goodput of this job.

    - `def throughput(self, num_nodes, num_replicas, atomic_bsz, accum_steps):`

        - Call `_predict_accum_time()` (for one-step per-GPU compute time, which is accum_time), `_predict_network_time()` (for network time), `_predict_log_optim_time()` (for total time, combine the two above).
        - **total time = accum_steps * accum_time + opt_time**;
            - Since each step gradient accumulation would also perform the fp and bp process, just not to perform weight update, **all the m+1 steps hold the same computing time**.
        - **opt_time = (accume_time ^ \gamma + network_time ^ {1 - \gamma}) ^ {1 / \gamma}**
        - Return batch size / total time = throughput.

    - `def efficiency(self, batch_size):`

        - Calculate statistical efficiency;

    - `def optimize(self, num_nodes, num_replicas, max_batch_size=None, atomic_bsz_range=None, accumulation=False):`

        - **PolluxAgent 收集 bs 和 $T_{iter}$ 等信息**，基于信息**获取 efficiency 和拟合 throughput 的函数**，进而**获取每个 job 的 goodput 函数**；通过**最大化 job 的 goodput 来动态调整 bs 和 lr（最大化操作通过多点 sample 并取最大来实现）**.

        - **Called in `./torch/data.py`**.

        - Reshape array:

            ```python
            # Remember what the output shape/format should be and flatten inputs.
            output_shape = np.broadcast(num_nodes, num_replicas).shape
            output_scalar = np.isscalar(num_nodes) or np.isscalar(num_replicas)
            num_nodes = np.broadcast_to(num_nodes, output_shape).flatten()
            num_replicas = np.broadcast_to(num_replicas, output_shape).flatten()
            ```

        -  Sample to simulate the goodput func:

            ```python
            # NOTE: Sample to simulate (fit) the goodput func
            
            # Samples 50 different total batch sizes in geometric space.
            min_batch_size = np.maximum(self._init_batch_size,
            min_atomic_bsz * num_replicas)
            batch_size = np.geomspace(min_batch_size, max_batch_size)               
            # NOTE: Default to 50 samples
            local_bsz = batch_size / num_replicas
            ```

            Then call the goodput evalutation func:

            ```python
            # Evaluate the goodput of all candidate configurations.
            goodput = self.evaluate(num_nodes, num_replicas, atomic_bsz, accum_steps)
            ```

            <font color=red> QUESTION: What is the exact formulation method of `num_nodes` and `num_replicas`?</font>

            - Scalar, but be broadcast to a array with only one element.

        - NOTE: When target local_bsz > max atomic bs, which means it cannot directly be processed  by the local deivice, we need to split and form the gradient accumulation to scale up the local bs.

- `def fit_perf_params(num_nodes, num_replicas, atomic_bsz, accum_step_time, optim_step_time):`

    - **PolluxAgent 对本地拟合 throughput 的参数进行更新，基于 profile 得到的真实执行时间数据**。
        - `./torch/_metrics.py` 里定义了 `_metric_state()` 并调用 `fit_perf_params()`，`./torch/metrics_test.py` 进行了 profile 的测试，`./torch/data.py` 内进行 runtime 时的 profile (`profile() in class AdaptiveDataLoaderHelper(object)`)；
    - Fit the performance model (**paramters include alpha and beta**) given accum time and optim time measurements for different configurations of num_nodes, num_replicas, and atomic_bsz.
    - Call `_obj_fn()`.
    - Locally use the `autograd.numpy` to use the differentiate func of `autograd`.

- `def _obj_fn(params, num_nodes, num_replicas, atomic_bsz, accum_step_time, optim_step_time):`

    - Call `_predict_accum_time()`, `_predict_log_optim_time()` and `_predict_network_time()` and calculate & return the error.

- **PolluxAgent 周期性向 PolluxSched 汇报 goodput func 是通过 PolluxSched 在 `./sched_adaptdl_sched/allocator.py` 中直接实例化 goodput 对象，将从 sched_hints 中得到的 params 和 init_bs 等 info 传给 goodput 对象，然后再调用 goodput 的成员函数进行 optmize 的**。

-----



#### 2. `./adaptdl/adaptdl/torch/_metrics.py`

- `class _MetricsState(adaptdl.checkpoint.State):`

    - `self.profile = collections.defaultdict(collections.Counter)`
        - To record the profile info.
    - `def save(self, fileobj)`
        - Save metrics states to fileobj.
    - `def load(self, fileobj)`
        - Load metrics states from fileobj.

- `def _metrics_state():`

    - The warpper function of metrics state class.

        ```python
        def _metrics_state():
            global _METRICS_STATE
            if _METRICS_STATE is None:
                _METRICS_STATE = _MetricsState()
                adaptdl.checkpoint.load_state(_METRICS_STATE)
            return _METRICS_STATE
        ```

- `def profile_step_start(atomic_bsz):`

    - Start the profile, set the atomic bs, record the time and set `state.sync_time = 0.0`.

- `def profile_sync_time(sync_time):`

    - Update the global sync time of state.
    - `_metrics_state().sync_time += sync_time`

- `def profile_step_commit(accumulation_step=False):`

    - **Get nodes and replicas num from the outer env** (env variable):

        ```python
        num_nodes = adaptdl.env.num_nodes()
        num_replicas = adaptdl.env.num_replicas()
        ```

    - **Update profile info**:

        ```python
        if accumulation_step:
        	state.profile[key]["accum_step_time"] += step_time
        	state.profile[key]["accum_count"] += 1
        else:
        	state.profile[key]["optim_step_time"] += step_time
        	state.profile[key]["optim_sync_time"] += state.sync_time
        	state.profile[key]["optim_count"] += 1
        ```

    - **Fit parameters and report schedule hints**:

        ```python
        if not accumulation_step:
        	if _PREV_REPORT is None:
          	_PREV_REPORT = time.time()
          if adaptdl.env.replica_rank() == 0 and time.time() - _PREV_REPORT > 30:
          	_fit_perf_params()
          	_report_sched_hints()
          	_PREV_REPORT = time.time()
        ```

- `def get_goodput_fn():`

    - Get the goodput function based on the current state.

        ```python
        state = _metrics_state()
        if state.grad_params is None or state.perf_params is None:
        	return None
        return GoodputFunction(state.perf_params, state.grad_params,
                                   state.init_batch_size)
        ```

- `def _fit_perf_params():`

    - A warpper function of `fit_perf_params()` in `./goodput.py`.

    - Get the profile info and convert to array.

    - Update & average to get the per-step time:

        ```python
        # NOTE: computing time = opt time - sync time
        # NOTE: opt count could be more than 1, since may perform several non-accumulation profile 
        # NOTE: To combine data points, classify the non-sync opt time to accum_step_time
        
        accum_step_time += optim_step_time - optim_sync_time
        accum_count += optim_count
        
        # NOTE Average to get the per-step time
        accum_step_time /= accum_count
        optim_step_time /= optim_count
        
        state.perf_params = fit_perf_params(num_nodes, num_replicas, atomic_bsz,
                                                accum_step_time, optim_step_time)
        ```

- `def _report_sched_hints():`

    - **Construct scheduling hint from metrics state and report to the PolluxSched**.

    - Call `post_sched_hints(sched_hints, adaptdl.env.job_id())` to post:

        - in `./adaptdl/sched_hints.py`, use `requests` package to send.
        - `adaptdl.env.job_id()` will return a unique job identifier or ``None`` by calling `os.getenv("ADAPTDL_JOB_ID")`.

    - Called by `profile_step_commit()`.

    - In `./sched_hints.py`:

        ```python
        SCHED_HINTS = MappingProxyType({'initBatchSize': 0,
                                        'localBszBounds': None,  # [min, max]
                                        'globalBatchSize': None,
                                        'maxBatchSize': 0,
                                        'maxProfiledReplicas': 0,
                                        'gradientAccumulation': False,
                                        'gradParams': None,
                                        'perfParams': None})
        ```

        **根据 profile 结果 states 来构造 scheduling hints！**

        **`initBatchSize` 等变量由用户指定，`gradParams` 和 `perfParams` 则由 local PolluxAgent 维护和不断更新！**

-----



#### 3. `./adaptdl/adaptdl/torch/accumulator.py`

- `class Accumulator(collections.abc.MutableMapping):`
    - This class helps aggregate simple statistics across all replicas in the current job, and across any number of checkpoint-restarts. Can be used to compute metrics like loss and accuracy, synchronized across each replica.

------



#### 4. `./adaptdl/adaptdl/torch/data.py`

- `class ElasticSampler(Sampler):`
    - 



Do the profile work
