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

1. How PolluxAgent (job-level) retrieve information?

    - 怎么实现：**PolluxAgent 收集 bs 和 $T_{iter}$ 等信息**，基于信息**获取 efficiency 和拟合 throughput 的函数**，进而**获取每个 job 的 goodput 函数**，在**动态调整 bs 和 lr 以获取最大的 job goodput**？

    - In `./goodput.py/class GoodputFunction`: 
        - **Call Flow**: `throughput()` & `efficiency()` $\Rightarrow$ `evaluate()` $\Rightarrow$ `optimize()`

2. How PolluxAgent: **基于 dataloader (`./torch/data.py`) 的 profile 结果**，**维护用于拟合 iteration time 和 throughput 的一系列参数**，并**周期性向 PolluxSched (cluster-level) 报告 goodput 函数和 scheduling hints**？

    - **Call Flow**: `_fit_perf_params()(fit_perf_params())` + `_report_sched_hints()` $\Rightarrow$ `profile_step_commit()` + `profile_step_start()` $\Rightarrow$ `profile()`

    - In `./goodput.py/def fit_perf_params()`: 
        - **PolluxAgent 对本地拟合 throughput 的参数进行更新，基于 profile 得到的真实执行时间数据**，给定不同 configurations of num_nodes, num_replicas, and atomic_bsz 下的 accum time and optim time measurements；


    - In `./torch/_metrics.py/def _fit_perf_params()`:
        - A warpper function of `fit_perf_params()` in `./goodput.py`, **Get the profile info and convert to array**, **update & average to get the per-step time**.


    - In `./torch/_metrics.py/def _report_sched_hints()`:
        - **Construct scheduling hint from metrics state and report to the PolluxSched**.

    - **PolluxSched 对 goodput 函数的调用是直接声明 goodput 类的 instance 来使用的**。

    - In `./torch/_metrics.py/def profile_step_start()`:
        - Start the profile in this step, called by dataloader in `./torch/data.py`.


    - In `./torch/_metrics.py/def profile_step_commit()`:
        - **Get nodes and replicas num from the outer env**, **update profile info into metrics state**, and **fit parameters and periodically report schedule hints**.
        - Called by dataloader in `./torch/data.py`.

    - In `./torch/data.py/class AdaptiveDataLoaderHelper(object)/def profile() `:
        - **Every iteration of every epoch should be profiled under this context**.
        - 是 **profile 的实际操作者**，**自己实现了 allreduce 等通信操作（用于同步 profile exit signal），用 socket 和 threading** (`./collective.py` -> `./reducer.py`)。


3. How PolluxSched (cluster-level): **基于 jobs 的 goodput 动态重分配资源**，通过**最大化 fitness 函数（由相较于 fair allocation 的 speedup 构造）来获取理论最优的分配**（指多 nodes 上的多 GPUs 到多 jobs 的映射），并**考虑多个集群层面的目标**，包括 fairness，goodput，reallocataion penalty，interference slowdown 等。

    - In `./sched/adaptdl_sched/allocator.py/class AdaptDLAllocator(object)`:

        - The **run function** of this class is:

            ```python
            async def run(self):
              await asyncio.gather(
                # 1) Watch for new job and start if possible.
                self._allocate_one_loop(),
                # 2) Periodically optimize existing jobs.
                self._optimize_all_loop()
              )
            ```

        - `self._allocate_one_loop()`: **Watch `kubernetes.watch.Watch().stream()` to watch and allocate events** (as jobs). 

        - `self._allocate_one(event)` : allocate one job, called by `self._allocate_one_loop()`. 

            - Read job from stream, parse job info, find available nodes, get allocation plan (calling `policy.allocate_job`). 

        - `self._optimize_all_loop()`: **Periodically access global lock and optimize existing jobs, then sleep for a time interval.**
        - `self._optimize_all()`: optimizing existing jobs allocation, called by `self._optimize_all_loop()`.
            - Find available nodes, get jobs and prev allocation, get allocation plan for all jobs considering prev allocations (calling `self._allocate()`), update current allocations.
        - `self._allocate()`: Get allocation plan for all jobs knowing nodes, jobs and prev allocation, called by `self._optimize_all()`.
            - Remove too-big jobs, try shrink the cluster (if need, expander), optimize the allocation (calling `policy.optimize()`), expand the cluster (if need).

    - In `./sched/adaptdl_sched/policy/pollux.py/class PolluxPolicy(object)`:

        - `def allocate_job(self, job_info, nodes):`
            - A simple strategy that **find the first available node for a new job**. 
        - `def optimize(self, jobs, nodes, base_allocations, node_template):`
            - **Sort jobs based on**: 1) **is_pinned state**; 2) **less min_replicas** (FIFO if the same); 3) **earlier creation timestamp**.
            - Sort nodes based on preemptible.
            - **Problem formulation and optimization**: [NSGA2 algorithm](https://baike.baidu.com/item/NSGA-Ⅱ/8524196?fr=aladdin) (多目标遗传算法) (In `class Problem(pymoo.core.problem.Problem)`, based on [pymoo: Multi-objective Optimization in Python](https://pymoo.org))
                - **Multi-objective optimization problem** used by PolluxPolicy to **determine resource allocations and desired cluster size**.  The **cluster performance** and **N** are the **two objectives being optimized**, resulting in **a set of Pareto-optimal solutions**.
                - **对 cluster autoscaling 的利用率设置上下阈值。** **Calculates the cluster utility for each state**, defined as **the average percentage of ideal speedup for each job (ie. speedup / num_replicas)**, **weighted by the job's share of the most congested cluster resource**. (**cluster util 的定义**).
                    - 在决定 nodes num 时尽可能使 cluster util 最优，进而在 `optimize()` 中决定 allocation (fairness).

4. How Pollux **interact with hardware resources (maybe K8S pods)**？

    - `./sched/adaptdl_sched/allocator.py/class AdaptDLAllocator(object)/async def _find_nodes():`

        - **Find available nodes**
        - Get node list, find all pods qualified by the pod_label_selector, update node info dict based on `node_list` and `pod_list` (calling `get_node_unrequested` in `resources.py`), construct node template and return.

    - `./sched/adaptdl_sched/allocator.py/class AdaptDLAllocator(object)/def _get_job_info():`

        - **Get jobs info**
        - Get resources of this job (calling `get_pod_requests()` in `resources.py`), Get scheduling hints, Construct speedup function of this job, etc. Return JobInfo object.

    - `./sched/adaptdl_sched/resources.py/def get_node_unrequested()`:

        - **Get the amount of node resources which are unrequested (还未被 pods 请求使用的，但后面可能会向 node 请求资源) by a list of pods**.

        - Args:

            - **node (kubernetes.client.V1Node)**: The node to get unrequested resources for.

            - **pods (List[kubernetes.client.V1Pod])**: Pods which may request resources from the node. 

        - Call `get_pod_requests()` and remove the already requested resources unit.

    - `./sched/adaptdl_sched/resources.py/def get_pod_requests()`: 

        - **Get the aggregate amount of resources requested by all containers in a pod**.

    - `./sched/adaptdl_sched/cluster_expander.py/class ClusterExpander(object):`

        - ClusterExpander tries to **keep expected node count available to the allocator**. It does that by **spawning equal number of placeholder pods (one of each node)**. **The pods have anti-affinity which prevents them from getting scheduled on the same node**. This pushes the cluster autoscaler to provision one node for each Pending placeholder.
        - Called by `allocator.py` based on the shrink trial and the optimization result.

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

        - **调用 `throughput()` 和 `efficiency()` 函数，构造获取每个 job 的 goodput 函数**。
        - Calculate overall batch size for calculating the statistical efficiency in `efficiency()`.

    - `def throughput(self, num_nodes, num_replicas, atomic_bsz, accum_steps):`

        - Call `_predict_accum_time()` (for one-step per-GPU compute time, which is accum_time), `_predict_network_time()` (for network time), `_predict_log_optim_time()` (for total time, combine the two above).
            - **收集 batch size 和 $T_{iter}$ 等信息**，基于信息**拟合 throughput 的函数**。
        - **total time = accum_steps * accum_time + opt_time**; 
            - Since each step gradient accumulation would also perform the fp and bp process, just not to perform weight update, **all the m+1 steps hold the same computing time**.
        - **opt_time = (accume_time ^ \gamma + network_time ^ {1 - \gamma}) ^ {1 / \gamma}**
        - Return batch size / total time = throughput.

    - `def efficiency(self, batch_size):`

        - Calculate statistical efficiency;
        - **基于 grad_params 和 batch size，计算 statistical efficiency**。

    - `def optimize(self, num_nodes, num_replicas, max_batch_size=None, atomic_bsz_range=None, accumulation=False):`

        - **根据 batch size 范围进行多次采样，模拟 function，并调用 evaluate() 函数计算相应的 goodput，返回最好的结果。**

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

            Then call the goodput evaluation func:

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
    - Fit the performance model (**paramters include alpha and beta**) given **accum time and optim time measurements for different configurations of num_nodes, num_replicas, and atomic_bsz**.
    - Call `_obj_fn()`.
    - Locally use the `autograd.numpy` to use the differentiate func of `autograd`.

- `def _obj_fn(params, num_nodes, num_replicas, atomic_bsz, accum_step_time, optim_step_time):`

    - Call `_predict_accum_time()`, `_predict_log_optim_time()` and `_predict_network_time()` and calculate & return the error.

- **PolluxAgent 周期性向 PolluxSched 汇报 goodput func 是通过 PolluxSched 在 `./sched_adaptdl_sched/allocator.py` 中直接实例化 goodput 对象，将从 sched_hints 中得到的 params 和 init_bs 等 info 传给 goodput 对象，然后再调用 goodput 的成员函数进行 optmize 的**。

- 注意，上面 PolluxSched 拿到的不一定是最新的 sched_hints，PolluxAgent 每隔一个 time interval (e.g., 30s) 更新一次 sched_hints：in `./_metrics.py/profile_step_commit()`:

    ```python
    if not accumulation_step:
    	if _PREV_REPORT is None:
    		_PREV_REPORT = time.time()
      
      # Time interval = 30
    	if adaptdl.env.replica_rank() == 0 and time.time() - _PREV_REPORT > 30:
    		_fit_perf_params()
    		_report_sched_hints()
    		_PREV_REPORT = time.time()
    ```

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
    - **主要面向 step time 从而计算 throughput**，未考虑 GPU utilization 等指标。
    - `def from_ray():` In `./env.py`
        - **Returns True if the code is being called from Ray**
        - <font color=red> checkpoint 的 load 和 save 均有 from ray 和 not from ray 两种情况，说明 adaptdl 可以集成到 ray 上</font>

- `def profile_sync_time(sync_time):`

    - Update the global sync time of state.
    - `_metrics_state().sync_time += sync_time`

- `def profile_step_commit(accumulation_step=False):`

    - **Get nodes and replicas num from the outer env** (env variable):

        ```python
        num_nodes = adaptdl.env.num_nodes()
        num_replicas = adaptdl.env.num_replicas()
        ```

    - **Update profile info into metrics state**:

        ```python
        if accumulation_step:
        	state.profile[key]["accum_step_time"] += step_time
        	state.profile[key]["accum_count"] += 1
        else:
        	state.profile[key]["optim_step_time"] += step_time
        	state.profile[key]["optim_sync_time"] += state.sync_time
        	state.profile[key]["optim_count"] += 1
        ```

    - **Fit parameters and periodically report schedule hints**:

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

    - **Get the profile info and convert to array**.

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
    - A PyTorch Sampler which partitions data samples across multiple replicas, and supports deterministic continuing across checkpoint-restarts. Shuffling is deterministic for each epoch.

- `def current_dataloader():`

    - Reference to the data loader currently being iterated:

        ```python
        return AdaptiveDataLoaderHelper._current
        ```

- `class AdaptiveDataLoaderHelper(object):`

    - This class provides fine-grained control over adaptive training loops. It can be used for building more user-friendly custom data loaders, such as `AdaptiveDataLoader`.

    - Arguments:

        - batch_size (int): The target total batch size across all replicas. The actual total batch size may be different due to rounding (each replica must have the same local batch size), or being scaled up using adaptive batch sizes.

    - **Class properties**:

        ```python
        # NOTE: The property of this class, not of a specified instance.
        
        # Epoch -> the number of dataloader loops completed so far in that epoch, across 
        # all AdaptiveDataLoader objects.
        _position = collections.Counter()
        _training = None  # The AdaptiveDataLoader which loads training data.
        _current = None  # The AdaptiveDataLoader which is currently iterating.
        ```

    - `def current_index(self):`

        - The total number of data samples processed so far in the current loop. Includes the data processed by all replicas. ``None`` if this data loader is not currently being iterated.

    - Is accumulation step or optimizing step?

        ```python
        def is_accum_step(self):
        	"""
        		Whether the current step's gradient will be accumulated.
        	"""
        	return self._accum_count < self._state.accumulation_steps
        
        def is_optim_step(self):
          """
          	Whether the optimizer step will be invoked in this step.
          """
          return not self.is_accum_step()
        ```

    - `def train(self):`

        - Set this data loader to be the one used for training. Only one data loader may be used for training.

    - `def autoscale_batch_size(self, max_batch_size, local_bsz_bounds=None, gradient_accumulation=False):`

        - Enables adaptive batch size. Should be invoked once after the data loader object is created.
        - Call `self.train()`.
        - `self.max_batch_size` is a indicator for whether applying batch size autoscale (if none, not apply).

    - `def _sync_local_bsz(self):`

        - **Decide the local batch size based on goodput function**, if not the first time, we calculate speedup function and decide whether to use the bs suggested by the new goodput function or not.

    - `def profile(self, commit):`

        - **Every iteration of every epoch should be profiled under this context**. Note that, custom DataLoader writers should make sure that it gets called equal number of times on each replica.
        - Arguments: `commit (bool)`: Whether to commit the profiled results.

        ```python
        # Synchronize the exit signal so all replicas exit after
        # the same iteration. Do this asynchronously to prevent
        # unnecessary blocking on the network.
        if self.future_exit is not None and self.future_exit.result():
        	adaptdl.checkpoint.save_all_states()
        	exit(143)  # Standard exit code response to SIGTERM.
        self.future_exit = adaptdl.collective.allreduce_async(get_exit_flag(), lambda a, b: a or b)
        profile_step_start(self.current_local_bsz)
        yield
        if commit:
        	profile_step_commit(self.is_accum_step())
        self._accum_count = (0 if self.is_optim_step() else self._accum_count + 1)
        ```

        - `yield` 的用法：
            - 和 `return` 的区别：
                - 有`return`的函数直接返回所有结果，程序终止不再运行，并销毁局部变量；
                - 有`yield`的函数则返回一个可迭代的 generator（生成器）对象，你可以使用for循环或者调用next() 方法遍历生成器对象来提取结果。
            - 在调用生成器函数的过程中，每次遇到 `yield` 时函数会暂停并保存当前所有的运行信息（保留局部变量），返回`yield`的值, 并在下一次执行`next()`方法时从当前位置继续运行，直到生成器被全部遍历完。
            - **在这里，应该是 `profile_step_start()` 内声明的 `state.step_start` 计时器被保存并持续运行，直到 `profile_step_commit()` 时再计算总的时间**。 注意，**下一次 `profile()` 函数再被调用时，一般是用 next() 来迭代该 generator，不会再执行一遍 `profile_step_start()`，因此不会覆盖 `state.step_start` 的值**。

        **自己实现了 allreduce 等通信操作，用 socket 和 threading** (`./collective.py` -> `./reducer.py`)。

        若 future_exit is not None 且结果已出 (.result() not None，说明其他 replicas 也完成了 profile 退出信号的生成)，保存 checkpoint 状态并 exit。否则，声明 exit signal future_exit 为 all reduce_async，退出信号的通信方式声明完成后，Call `profile_step_start()` 来记录 step_start 开始的时间，yield 会保存 state.step_start 的值，每次**再次调用 profile 时会直接从 yield 后面的代码开始执行**，即判断是否要 commit。**一般来说，start 后面都会接一个 commit，除非这轮 profile 数据需要丢弃，否则不管是 accum step 还是 opt step 都要这样**。

- `class AdaptiveDataLoaderMixin(object):`

    - This class **provides elastic functionality to any custom DataLoader which inherits it**. It **defines a member _elastic of class `AdaptiveDataLoaderHelper` which has useful methods and members to implement restart-safe, elastic DataLoaders**. It also exposes public methods which can be used inside training loops directly from class `AdaptiveDataLoader`.

- `class AdaptiveDataLoader(DataLoader, AdaptiveDataLoaderMixin):`

    ```python
    """
        This class is a PyTorch DataLoader that also supports adaptive batch sizes
        and checkpoint-restart elasticity. Applications can typically use objects
        of this class as direct replacements for PyTorch DataLoaders. However, some
        notable differences are:
    
        1.  The ``batch_size`` argument defines the target total batch size across
            all replicas, rather than the local batch size on each replica.
        2.  Custom ``sampler`` and ``batch_sampler`` are not supported.
        3.  Iterating through the dataloader is only allowed from within an epoch
            loop (see :mod:`adaptdl.torch.epoch`), and only one dataloader loop is
            allowed at any given time.
       
        Arguments:
            dataset (torch.util.data.Dataset): Dataset from which to load the data.
            batch_size (int): The target total batch size across all replicas. The
                actual total batch size may be different due to rounding (each
                replica must have the same local batch size), or being scaled up
                using adaptive batch sizes.
            shuffle (bool): Whether the data is reshuffled at every epoch.
            **kwargs: Keyword arguments passed to ``torch.util.data.Dataloader``.
    
        Raises:
            ValueError: If ``sampler`` or ``batch_sampler`` are not ``None``.
    
        .. automethod:: __iter__
    """
    ```

- `class _AdaptiveDataLoaderState(adaptdl.checkpoint.State):`
    - Assume dataloaders are initialized in the same order in every replica. Keep a map of epoch -> number of dataloaders initialized so far in that epoch, and use that count to construct a unique name for the state.

------



#### 5. `./sched/adaptdl_sched/allocator.py`

- `class AdaptDLAllocator(object):`

    - **Init function**:

        ```python
        def __init__(self, expander):
                self._core_api = kubernetes.client.CoreV1Api()
                self._objs_api = kubernetes.client.CustomObjectsApi()
                self._custom_resource = ("adaptdl.petuum.com", "v1",
                                         "", "adaptdljobs")
                self._cluster_expander = expander
                self._policy = PolluxPolicy()
                # lock for the two corountines in run()
                self._lock = asyncio.Lock()
        ```

    - **Run function**:

        ```python
        async def run(self):
                # two functionality: (1) watch for new job and start if possible.
                # (2) periodically optimize existing jobs
                await asyncio.gather(
                    self._allocate_one_loop(),
                    self._optimize_all_loop()
                )
        ```

    - **Watch for new jobs and start if possible**:

        ```python
        async def _allocate_one_loop(self):
        	async with kubernetes.watch.Watch() as watch:
            while True:
              async for event in watch.stream(
                self._objs_api.list_namespaced_custom_object,
                *self._custom_resource, timeout_seconds=60):
                # We only consider newly-added preemptible jobs
                # because this allocation may not be final.
                if (event["type"] == "ADDED" and
                    event["object"]["spec"].get("preemptible", True)):
                  async with self._lock:
                    await self._allocate_one(event)
        ```

        **从 K8S 的 watch stream 中监听 events**.

        `async def _allocate_one(self, event):`

        - Read the job from `kubernetes.client.CustomObjectsApi().list_namespaced_custom_object`
        - Parse the job info: `job_info = self._get_job_info(job)`
        - Find available nodes: `node_infos, _ = await self._find_nodes()`
        - Get allocation plan for new jobs: `new_allocation = self._policy.allocate_job(job_info, node_infos)`
            - `policy.allocate_job()` is improted from `./policy/pollux.py`
        - Patch job status: `await patch_job_status(self._objs_api, namespace, name, patch)`

    - **Periodically optimize existing jobs**:

        ```python
        async def _optimize_all_loop(self):
                while True:
                    # try to gain lock
                    async with self._lock:
                        await self._optimize_all()
        
                    LOG.info("Sleep for 60 seconds")
                    await asyncio.sleep(60)
        ```

        **Gain the global lock and perform optimize all, then sleep for a time interval.**

        `async def _optimize_all(self):`

        - Find available nodes: `nodes, node_template = await self._find_nodes(pod_label_selector="!adaptdl/job")`

        - Get jobs and prev allocations: `jobs, prev_allocations = await self._find_jobs_and_allocations()`

        - Get allocation plan for all jobs considering prev allocations: `allocations = self._allocate(jobs, nodes, prev_allocations, node_template)`

        - Update current allocations: `await self._update_allocations(allocations)`

            - Get job list: `job_list = await self._objs_api.list_namespaced_custom_object("adaptdl.petuum.com", "v1", "", "adaptdljobs")`

            - Update job info:

                ```py
                for job in job_list["items"]:
                	namespace = job["metadata"]["namespace"]
                	name = job["metadata"]["name"]
                	job_allocation = job.get("status", {}).get("allocation", [])
                	new_allocation = list(allocations.get((namespace, name), []))
                	if list(job_allocation) != new_allocation:
                		patch = {"status": {"allocation": new_allocation}}
                		LOG.info("Patch AdaptDLJob %s/%s: %s", namespace, name, patch)
                		await patch_job_status(self._objs_api, namespace, name, patch)
                ```

    - **Find available nodes:** `async def _find_nodes(self, pod_label_selector=""):`

        - Get node list: `node_list = await self._core_api.list_node()`

        - Find all pods qualified by the pod_label_selector:

            ```python
            pod_list = await self._core_api.list_pod_for_all_namespaces(
                                            label_selector=pod_label_selector)
            ```

        - Update node info dict based on `node_list` and `pod_list`:

            ```python
            for node in node_list.items:
            	if allowed_taints(node.spec.taints):
            		resources = get_node_unrequested(node, pod_list.items)
            		if not resources.get("pods"):
            			LOG.warning(f"node {node.metadata.name} "
                              "has no free pods available.")
            		node_infos[node.metadata.name] = NodeInfo(resources, False)
            ```

            **`get_node_unrequested()` 定义在 `./resources.py` 中，是对 node 和 pod 的抽象，即对底层 K8S pod 进行抽象。**

        - Construct node template and return:

            ```python
            max_resources = {}
            for node_name in node_infos:
            	for key, val in node_infos[node_name].resources.items():
            		if key not in max_resources or val > max_resources[key]:
            			max_resources[key] = val
            node_template = NodeInfo(max_resources, True)
            
            return node_infos, node_template
            ```

            We **infer each resource to be the maximum amount observed in any real node. 不同状态下 real node 内的 pods 数目可能不同**。 

    - **Get jobs info**: `def _get_job_info(self, job):`

        - Get resources of this job: `resources = get_pod_requests(job["spec"]["template"]["spec"])`

            - `get_pod_requests()` is from `./resources.py`

        - Get scheduling hints: `hints = job.get("status", {}).get("train", {})`

        - Construct speedup function of this job:

            - Construct goodput instance and pass in the params info based on scheduling hints:

                ```python
                goodput_fn = GoodputFunction(perf_params, grad_params,
                                                         hints["initBatchSize"])
                ```

            - Construct speedup function based on goodput function and scheduling hints:

                ```python
                speedup_fn = SpeedupFunction(
                                goodput_fn,
                                hints.get("maxBatchSize"),
                                hints.get("localBszBounds"),
                                hints.get("gradientAccumulation", False))
                ```

        - Get max and min replicas of this job based on job info and hints.

        - Get `preemptible` based on job info.

        - Return: 

            ```python
            return JobInfo(
                            resources, speedup_fn, creation_ts, min_replicas,
                            max_replicas, preemptible)
            ```

    - **Get jobs and prev allocations**: `async def _find_jobs_and_allocations(self):`

        - Get job list: `job_list = await self._objs_api.list_namespaced_custom_object("adaptdl.petuum.com", "v1", "", "adaptdljobs")`
        - Construcr job info and allocations based on job list.

    - **Get allocation plan for all jobs considering prev allocations**: `def _allocate(self, jobs, nodes, prev_allocations, node_template):`

        - Remove jobs when no node can fit a replica of this job.

        - If there are no jobs, let the expander shrink the cluster:

            ```python
            self._cluster_expander.fit([])
            ```

        - **Optimize the allocation**: call `optimize()` from `./policy/pollux/PolluxPolicy class`

            ```py
            allocations, desired_nodes = self._policy.optimize(
                            jobs, nodes, prev_allocations, node_template)
            ```

        - **Expand the cluster**: 

            ```python
            if desired_nodes < len(nodes):
            	active_nodes = list(set.union(*map(set, allocations.values())))
            else:
            	active_nodes = list(nodes)
            	while len(active_nodes) < desired_nodes:
            		active_nodes.append(f"~{desired_nodes-len(active_nodes)}")
                        self._cluster_expander.fit(active_nodes)
            ```

- **Main function**:

    ```python
    if __name__ == "__main__":
        logging.basicConfig()
        kubernetes.config.load_incluster_config()
    
        expander = ClusterExpander()
        allocator = AdaptDLAllocator(expander)
    
        loop = asyncio.get_event_loop()
        loop.run_until_complete(asyncio.gather(
            expander.run(),
            allocator.run(),
        ))
        loop.close()
    ```

-----



#### 6. `./sched/adaptdl_sched/cluster_expander.py`

- `class ClusterExpander(object):`

    ClusterExpander tries to **keep expected node count available to the allocator**. It does that by **spawning equal number of placeholder pods (one of each node)**. **The pods have anti-affinity which prevents them from getting scheduled on the same node**. This pushes the cluster autoscaler to provision one node for each Pending placeholder. **The job of ClusterExpander then is really to maintain the requested count of placeholders**. This also holds when the jobs are scaling down or finishing and we no longer need specific nodes.

    - Init function:

        ```python
        self._v1 = kubernetes.client.CoreV1Api()
        self._apps_api = kubernetes.client.AppsV1Api()
        self._allocations = set()
        self._active_nodes = set()
        ```

    - **Generate placeholder pod**: `def _gen_placeholder_pod(self):`

    - `async def _reconcile(self, expected):`

        - Get & sort pods:

            ```python
            ret = await self._v1.list_namespaced_pod(config.get_namespace(),
            								label_selector=f{config.ADAPTDL_PH_LABEL}=true")  # noqa: E501
            
            pods = [Pod(pod.metadata.name, pod.spec.node_name, pod.status.phase) for pod in ret.items]
            
            # sort pods based on 1. in allocations 2. Running 3. Pending
            pods = sorted(pods, key=key_fn)
            ```

        - Recheck whether the generated pods are sufficient or redundant:

            ```python
            if expected == spawned:
                            return
                        elif expected > spawned:  # spawn the difference
                            r = []
                            for _ in range(expected - spawned):
                                r.append(self._v1.create_namespaced_pod(
                                    body=self._gen_placeholder_pod(),
                                    namespace=config.get_namespace()))
                            ret = await asyncio.gather(*r)
                            for pod in ret:
                                LOG.info(f"Spawned {pod.metadata.name}")
                        else:   # expected < spawned, delete from the left
                            r = []
                            for name in [x.name for x in pods][:spawned - expected]:
                                assert name not in self._allocations
                                r.append(self._v1.delete_namespaced_pod(
                                    name=name, namespace=config.get_namespace()))
                            await asyncio.gather(*r)
                            for name in [x.name for x in pods][:spawned - expected]:
                                LOG.info(f"Deleted {name}")
            ```

    - Run function:

        ```python
        async def run(self):
                adaptdl_sched = \
                    await self._apps_api.read_namespaced_deployment(
                        namespace=config.get_namespace(),
                        name=config.get_adaptdl_deployment())
                self._owner_reference = templates.owner_reference_template(
                    config.get_namespace(),
                    adaptdl_sched.metadata.name,
                    adaptdl_sched.metadata.uid,
                    "Deployment",
                    "apps/v1")
                while True:
                    await self._reconcile(len(self._active_nodes))
                    await asyncio.sleep(30)
        ```

    - Fit function:

        ```python
        def fit(self, active_nodes):
                """ active_nodes contain allocations + virtual nodes, our job at the
                expander is to 1. maintain the allocations, and 2. provision nodes in
                active nodes and not in allocations."""
                self._active_nodes = set(active_nodes)
                self._allocations = set()
                for node in self._active_nodes:
                    if not node.startswith("~"):  # real nodes only
                        self._allocations.add(node)
        ```

        需要用传进来的 `active_nodes` 参数来更新 `self._active_nodes`，进而更新 `self._allocations`。

    - Main function:

        ```python
        if __name__ == '__main__':
            # unit test to check basic sanity
            loop = asyncio.get_event_loop()
            loop.run_until_complete(kubernetes.config.load_kube_config())
        
            expander = ClusterExpander()
        
            async def run():
                expander.fit(['n0', '~n1'])
                await expander.run()
        
            loop.run_until_complete(run())
            loop.close()
        ```

        **先 fit，再 run ClusterExpander。**

------



#### 7. `./sched/adaptdl_sched/controller.py`

- `class AdaptDLController(object):`
    - The main controller **responsible for the overall AdaptDLJob lifecycle.** 
    - Essentially, it **keeps a queue of AdaptDLJobs whose states may need to be synchronized**. 
    - It **watches for events such as pod status changes and allocation changes and enqueues any potentially affects AdaptDLJobs**. 
    - **A worker coroutine is responsible for processing AdaptDLJobs from the queue** and **guarantees that a single AdaptDLJob is never processed concurrently**.

-----



#### 8. `./sched/adaptdl_sched/resources.py`

- `def get_node_unrequested(node: kubernetes.client.V1Node, pods: List[kubernetes.client.V1Pod]) -> Dict[str, int]:`

    - **Get the amount of node resources which are unrequested (还未被 pods 请求使用的，但后面可能会向 node 请求资源) by a list of pods**.

    - Args:

        - **node**: The node to **get unrequested resources for**.

        - **pods**: Pods which **may request resources from the node**.

    - Returns:

        - **Mapping from resource names (eg. cpu, memory, nvidia.com/gpu)** to **an integer amount of the resource which is unrequested on the node**. The integer amounts are discretized to the smallest possible unit of each resource.

    - Call `get_pod_requests(())` and remove the already requested resources unit.

- `def get_pod_requests(pod_spec: Union[kubernetes.client.V1PodSpec, dict]) -> Dict[str, int]:`
    - **Get the aggregate amount of resources requested by all containers in a pod**.
    - Args:
        - **pod_spec**: **The pod to get requested resources for**.
    - Returns:
        - **Mapping from resource names (eg. cpu, memory, nvidia.com/gpu) to an integer amount of the resource which is requested by the pod**. The integer amounts are discretized to the smallest possible unit of each resource.
- `def set_default_resources(pod_spec: dict) -> dict:`
    - **Set the default resources for a given AdaptDLJob spec**.
    - Args:
        - **pod_spec: The pod spec to set default resources for**.
    - Returns:
        - **A new pod spec with default resources set**.
- `def _discretize_resource(name, value):`
    - Normalize to the smallest integral units.

--------



#### 9. `./sched/adaptdl_sched/supervisor.py`

- `class Supervisor:`

    - Supervisor **provides a simple REST interface for several functionalities**. Currently, it has **two endpoints**:

        - **/hints for jobs to send scheduling hints**.

        - **/discover for finding the pod IPs of a job**.

-----------



#### 10. `./sched/adaptdl_sched/policy/speedup.py`

- `class SpeedupFunction(object):`
    - Construct a speedup function for a job, use `__call(self, num_nodes, num_replicas)__` method to call as a function.
    - Init with `goodput_fn, max_batch_size=None, atomic_bsz_range=None, accumulation=False, mem_size=32`.

-------



#### 11. `./sched/adaptdl_sched/policy/pollux.py`

**The problem formulation and optimization are based on [pymoo: Multi-objective Optimization in Python](https://pymoo.org)**.

- `class PolluxPolicy(object):`

    - Init function:

        ```python
        def __init__(self):
                self._prev_states = None
                self._prev_jobs = None
                self._prev_nodes = None
                # Utilization thresholds for cluster autoscaling.
                self._min_util = 0.35
                self._max_util = 0.65
        ```

        **对 cluster autoscaling 的利用率设置上下阈值。**

    - `def allocate_job(self, job_info, nodes):`

        - A simple strategy that **find the first available node for a new job**. This method is intended to **allocate a single arriving job**. It **expects the node resources to take into account adaptdl and non-adaptdl pods**.

            - <font color=red>什么叫需要 node resources 同时考虑 adaptdl 和 non-adaptdl pods? 可能的原因是，仅一个新来的 job 进行 allocate, 此时集群内同时存在 adaptdl 和 non-adaptdl pods，因此需要同时考虑。</font>

        - Arguments:

            - **job_info (JobInfo): JobInfo object of the job**

            - **nodes (dict): dict from node name to node_info**

        - Returns:

            - list(str): allocation of the job, e.g. [node name 0, node name 1, ...] if found available node, else an empty list.

        ```python
        for node_name, node in nodes.items():
                    # number of replica fit in this node
                    replica_this = min(node.resources.get(key, 0) // val
                                       for key, val in job_resources.items())
        
                    # NOTE: All target nodes are the same one.
        
                    if replica_this >= min_replicas:
                        node_list = [node_name] * min_replicas
                        return node_list
                else:
                    return []
        ```

    - **Allocation -> state**:

        ```python
        # NOTE: The initialization of state variable: state = np.zeros(len(jobs), len(nodes))
        
            def _allocations_to_state(self, allocations, jobs, nodes):
                jobs_index = {key: idx for idx, key in enumerate(jobs)}
                nodes_index = {key: idx for idx, key in enumerate(nodes)}
                state = np.zeros((len(jobs), len(nodes)), dtype=np.int)
                for job_key, alloc in allocations.items():
                    for node_key in (key for key in alloc if key in nodes_index):
                        state[jobs_index[job_key], nodes_index[node_key]] += 1
                return state
        ```

    - **State -> allocation**:

        ```python
        def _state_to_allocations(self, state, jobs, nodes):
                allocations = {}
                for job_idx, job_key in enumerate(jobs):
                    for node_idx, node_key in enumerate(nodes):
                        count = state[job_idx, node_idx]
                        allocations.setdefault(job_key, []).extend([node_key] * count)
                return allocations
        ```

    - `def _adapt_prev_states(self, jobs, nodes):`

        - **Adapt the previously saved optimization states to initialize the current genetic algorithm states**.

    - **Get the desired nodes based on utilization and the previous node sort result**:

        ```python
        def _desired_nodes(self, utilities, values, nodes):
          # NOTE: In case that allocating len(node) nodes is the best solution
          idx = self._select_result(values, len(nodes))
          if idx is not None and \
          								self._min_util <= utilities[idx] <= self._max_util:
            return len(nodes)
        
          # NOTE: Target util
          target_util = (self._min_util + self._max_util) / 2
        
          # NOTE: This is based on the fact that allocating len(node) nodes is not the best solution
          best_util = np.inf
          best_nodes = len(nodes)
        
          for util, (_, num_nodes) in zip(utilities, values):
            if util < self._min_util:
              continue
        
            if np.isclose(util, best_util) and num_nodes > best_nodes:
              # NOTE: This is because the util and best_util is roughly the same, we prefer more desired nodes
              best_nodes = num_nodes
        
            if abs(util - target_util) < abs(best_util - target_util):
              # NOTE: If current util is closer to the target util, choose it as the best nodes num
              best_util = util
              best_nodes = num_nodes
        
          return int(best_nodes)
        ```

    - `def optimize(self, jobs, nodes, base_allocations, node_template):`

        - **Run one optimization cycle of the Pollux scheduling policy**. This method **expects the node resources to only take into account non-adaptdl pods**.

            - <font color=red>什么叫需要 node resources 仅考虑 non-adaptdl pods? 可能的原因是，所有未完成的 jobs 都进行 reschedule，因此不存在 adaptdl-pods 了</font>

        - Arguments:

            - **jobs (dict): map from job keys to `JobInfo` objects which correspond to the incomplete jobs which should be optimized.** (未完成需要优化的 jobs)

            - **nodes (dict): map from node keys to `NodeInfo` objects which correspond to the existing nodes in the cluster.** (集群内已有的 nodes)

            - **base_allocations (dict): map from job keys to their current resource allocations, in the form of a list of a node key for each replica.**

            - **node_template (NodeInfo): represents a node which can be requested, used to decide the cluster size for cluster auto-scaling.** (用来做 cluster auto-scale 的样板 nodeInfo)

        - Returns:

            - **dict: map from job keys to their optimized resource allocations**, in the form of a list of a node key for each replica.

        - **Pinned job**：A job is considered pinned if it's non-preemptible and already has an allocation.

        - **Sort jobs based on**: 1) **is_pinned state**; 2) **less min_replicas** (FIFO if the same); 3) **earlier creation timestamp**.

            ```python
            # We sort the jobs based on min_replicas and then creation_timestamp,
            # so jobs wanting lower or no min_replicas guarantees are prioritized
            # ahead of those wanting higher min_replicas guarantees to avoid
            # underutilization of cluster. Within a same min_replicas value, they
            # will follow FIFO order. Pinned jobs are aggregated at front because
            # they already have an allocation and won't affect allocations of the
            # rest of the jobs.
            
            jobs = OrderedDict(sorted(jobs.items(), key=lambda kv: (not ispinned(kv[0], kv[1]), kv[1].min_replicas, kv[1].creation_timestamp)))
            ```

        - Sort nodes based on preemptible:

            ```python
            nodes = self._sort_nodes(nodes)
            ```

        - **Problem formulation and optimization**: [NSGA2 algorithm](https://baike.baidu.com/item/NSGA-Ⅱ/8524196?fr=aladdin) (多目标遗传算法)

            ```python
            problem = Problem(list(jobs.values()), list(nodes.values()) +
                                      len(nodes) * [node_template], base_state)
            algorithm = NSGA2(
                        pop_size=100,
                        # pymoo expects a flattened 2-D array.
                        sampling=states.reshape(states.shape[0], -1),
                        crossover=Crossover(),
                        mutation=Mutation(),
                        repair=Repair(),
                    )
            result = pymoo.optimize.minimize(problem, algorithm, ("n_gen", 100))
            states = result.X.reshape(result.X.shape[0], len(jobs), 2 * len(nodes))
            ```

            <font color=red>优化 metrics 是什么？在 Problem Class 中找。</font>

        - Construct return vars:

            ```python
            # Construct return values.
            utilities = problem.get_cluster_utilities(states)
            desired_nodes = self._desired_nodes(utilities, values, nodes)
            # NOTE: idx is the best choice idx in multiple states
            idx = self._select_result(values, min(len(nodes), desired_nodes))
            
            return (self._state_to_allocations(states[idx], jobs, nodes)
                            if idx is not None else {}), desired_nodes
            ```

- `class Problem(pymoo.core.problem.Problem):`

    - **Multi-objective optimization problem** used by PolluxPolicy to **determine resource allocations and desired cluster size**. Optimizes for the best performing cluster allocation **using only the first N nodes**. The **cluster performance** and **N** are the **two objectives being optimized**, resulting in **a set of Pareto-optimal solutions**.

    - 由于继承了 `pymoo.core.problem.Problem`，因此 `minimize` 等函数不用自己实现。

    - **The optimization states are a 3-D array of replica assignments with shape (pop_size x num_jobs x num_nodes)**. The element at k, j, n **encodes the number of job j replicas assigned to node n, in the kth solution**.

    - Arguments:

        - **jobs (list)**: list of JobInfo objects describing the incomplete jobs which need to be scheduled.

        - **nodes (list)**: list of NodeInfo objects describing the nodes in the cluster, **in decreasing order of allocation preference**.
        - **base_state (numpy.array)**: base optimization state corresponding to the current cluster allocations. Shape: (num_jobs x num_nodes).

    - **Init the super class**: `super().__init__(n_var=self._base_state.size, n_obj=2, type_var=np.int)`

    - `def _get_avail_resource(self, node_idx, node, rtype):`

        - Cutoff node's maximum allowable resources by amount already used by pinned jobs.
        - **Used for determining the upper bound of max replicas on this node for jobs.**

    - `def get_cluster_utilities(self, states):`

        - **Calculates the cluster utility for each state**, defined as **the average percentage of ideal speedup for each job (ie. speedup / num_replicas)**, **weighted by the job's share of the most congested cluster resource**. (**cluster util 的定义**)
        - Arguments:
            - **states (numpy.array)**: a **(pop_size x num_jobs x num_nodes) array** containing the assignments of job replicas to nodes.
        - Returns:
            - **numpy.array**: a **(pop_size) array containing the utility for each state**.

    - `def _crossover(self, states, **kwargs)`, `def _mutation(self, states, **kwargs)`, `def _repair(self, pop, **kwargs)`: 

        - 自定义遗传算法的交叉、变异和修复 method.
