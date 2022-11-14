Title: Encounter a 'profile worker forcely killed' & NotImplementedError (auto-slicing layers with existing physical meshes) when trying to enlarge the global batch size.



Hi, there. I'm trying to run Alpa PipeShardParallel training for a ResNet50 model (CIFAR10 dataset) on a Distributed Ray Cluster with two 1080ti (4-GPU each) servers. 

- The **parallel method** is as follows:

```python
method = alpa.PipeshardParallel(num_micro_batches=self.num_micro_batches,                                   										layer_option=alpa.AutoLayerOption(layer_num=self.pipeline_layer_num),
                                        stage_option="auto")
```

- The **training configs** are as follows:

```python
trainer_cfgs = {
        'model_name': 'wide_resnet',
        'dataset_name': 'CIFAR10',
        'batch_size': 16,
        'lr': 1e-3,
        'momentum': 0.9,
        'rand_seed': 123,
        'dtype': jnp.float32,
        'num_micro_batches': 16,
        'pipeline_layer_num': 16,
        'parallel_mode': 'search',
        'niter': 12,
        'profile_driver_time': True,
    }
```

- The **topology of the Ray cluster** is:

1080ti-01 (head node, 4 GPUs) -> 1080ti-02 (worker node, 4 GPUs).

- The **Alpa cluster is initialized** as:

```python
ray.init(address="auto")
alpa.init(cluster="ray", num_devices_per_node=2, num_nodes=2)
```



As shown above, when I set the batch size to 16, everything went well and I can finish the training process. However, when I **enlarged the batch size to 64**, the following errors occurred:

- **When profiling submesh (1, 1)** (the above profile for (2, 2) and (1, 2) went well):

    ```bash
    - Profiling for submesh 0 (1, 1):
    - Generate all stage infos (Jaxpr -> HLO)
    - Compile all stages
    - Profile all stages
    WARNING:alpa.pipeline_parallel.stage_profiling:Meet unexpected error, all profile workers are forcely killed
    (ProfileWorker pid=556309, ip=10.2.64.52) E1114 13:52:04.551004889  556309 server_chttp2.cc:40]        {"created":"@1668405124.550972237","description":"No address added out of total 1 resolved","file":"external/com_github_grpc_grpc/src/core/ext/transport/chttp2/server/chttp2_server.cc","file_line":395,"referenced_errors":[{"created":"@1668405124.550969355","description":"Unable to configure socket","fd":57,"file":"external/com_github_grpc_grpc/src/core/lib/iomgr/tcp_server_utils_posix_common.cc","file_line":216,"referenced_errors":[{"created":"@1668405124.550963496","description":"Cannot assign requested address","errno":99,"file":"external/com_github_grpc_grpc/src/core/lib/iomgr/tcp_server_utils_posix_common.cc","file_line":189,"os_error":"Cannot assign requested address","syscall":"bind"}]}]}
    (ProfileWorker pid=556309, ip=10.2.64.52) 2022-11-14 13:52:04,553	ERROR worker.py:451 -- Exception raised in creation task: The actor died because of an error raised in its creation task, ray::ProfileWorker.__init__() (pid=556309, ip=10.2.64.52, repr=<alpa.pipeline_parallel.stage_profiling.ProfileWorker object at 0x7fbcce45ca00>)
    (ProfileWorker pid=556309, ip=10.2.64.52)   File "/home/cyxue/Projects/playground/alpa/alpa/alpa/pipeline_parallel/stage_profiling.py", line 251, in __init__
    (ProfileWorker pid=556309, ip=10.2.64.52)     self.mesh = virtual_mesh.get_physical_mesh()
    (ProfileWorker pid=556309, ip=10.2.64.52)   File "/home/cyxue/Projects/playground/alpa/alpa/alpa/device_mesh.py", line 1900, in get_physical_mesh
    (ProfileWorker pid=556309, ip=10.2.64.52)     self.launched_physical_mesh = DistributedPhysicalDeviceMesh(
    (ProfileWorker pid=556309, ip=10.2.64.52)   File "/home/cyxue/Projects/playground/alpa/alpa/alpa/device_mesh.py", line 974, in __init__
    (ProfileWorker pid=556309, ip=10.2.64.52)     self.service_server, self.workers = self.launch_xla_servers()
    (ProfileWorker pid=556309, ip=10.2.64.52)   File "/home/cyxue/Projects/playground/alpa/alpa/alpa/device_mesh.py", line 1001, in launch_xla_servers
    (ProfileWorker pid=556309, ip=10.2.64.52)     service_server = xla_client._xla.get_distributed_runtime_service(
    (ProfileWorker pid=556309, ip=10.2.64.52) jaxlib.xla_extension.XlaRuntimeError: UNKNOWN: Failed to start RPC server
    ```

- After that, **the driver can 'correctly' obtain the optimized mesh results** as:

    ```python
    Result forward_stage_layer_ids: [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]
    Result mesh_shapes: [(1, 2), (1, 2)]
    Result logical_mesh_shapes: [(1, 2), (1, 2)]
    Result autosharding_option_dicts: [{'force_batch_dim_to_mesh_dim': 0}, {'force_batch_dim_to_mesh_dim': 0}]
    (MeshHostWorker pid=1706083) 1080ti-01:1706083:1706083 [1] NCCL INFO Bootstrap : Using eno2:10.2.64.51<0>
    (MeshHostWorker pid=1706083) 1080ti-01:1706083:1706083 [1] NCCL INFO NET/Plugin : No plugin found (libnccl-net.so), using internal implementation
    (MeshHostWorker pid=1706083) 1080ti-01:1706083:1706083 [0] NCCL INFO cudaDriverVersion 11070
    (MeshHostWorker pid=1706083) NCCL version 2.14.3+cuda11.7
     - Compile (driver): 1464.64 s
    compilation time breakdown: {'stage-construction': '1440.34', 'stage-construction-dp': '2.01', 'stage-construction-compilation': '443.24', 'stage-construction-profiling': '994.08'}
     - Compile (worker): 15.50 s
    [I] Training process warmup with dummy input batch...
    
    ########### [Init with NCCL] ###############
    ... ...
    ############################################
    
    [I] Ready to perform training process.
    [I] Batch (iteration) num 783 | Batched data shape: (64, 32, 32, 3) | Batched labels shape: (64,)
    [I] Benchmark the training process with entire dataset and profile with driver overhead...
        - Iteration 500 / 783 is performed...
    ```

- At this moment, **the `NotImplementedError` occurs**:

    ```bash
    Traceback (most recent call last):
      File "train.py", line 368, in <module>
        trainer.train()
      File "train.py", line 295, in train
        executable) = compile_and_benchmark_pipeshard_training_executable(
      File "/home/cyxue/Projects/playground/slice_profile/jax/benchmark_parallel_utils.py", line 410, in compile_and_benchmark_pipeshard_training_executable
        latencies, e2e_total_time, niter, local_lats = benchmark_training_executable(
      File "/home/cyxue/Projects/playground/slice_profile/jax/benchmark_parallel_utils.py", line 268, in benchmark_training_executable
        state = train_step(state, batches[i])
      File "/home/cyxue/miniconda3/envs/alpa/lib/python3.8/site-packages/jax/_src/traceback_util.py", line 162, in reraise_with_filtered_traceback
        return fun(*args, **kwargs)
      File "/home/cyxue/Projects/playground/alpa/alpa/alpa/api.py", line 121, in __call__
        self._decode_args_and_get_executable(*args))
      File "/home/cyxue/Projects/playground/alpa/alpa/alpa/api.py", line 191, in _decode_args_and_get_executable
        executable = _compile_parallel_executable(f, in_tree, out_tree_hashable,
      File "/home/cyxue/miniconda3/envs/alpa/lib/python3.8/site-packages/jax/linear_util.py", line 295, in memoized_fun
        ans = call(fun, *args)
      File "/home/cyxue/Projects/playground/alpa/alpa/alpa/api.py", line 218, in _compile_parallel_executable
        return method.compile_executable(fun, in_tree, out_tree_thunk,
      File "/home/cyxue/Projects/playground/alpa/alpa/alpa/parallel_method.py", line 233, in compile_executable
        return compile_pipeshard_executable(
      File "/home/cyxue/Projects/playground/alpa/alpa/alpa/pipeline_parallel/compile_executable.py", line 92, in compile_pipeshard_executable
        pipeshard_config = compile_pipeshard_executable_internal(
      File "/home/cyxue/Projects/playground/alpa/alpa/alpa/pipeline_parallel/compile_executable.py", line 177, in compile_pipeshard_executable_internal
        manual_stage_option) = cluster_layers_and_slice_mesh(
      File "/home/cyxue/Projects/playground/alpa/alpa/alpa/pipeline_parallel/stage_construction.py", line 734, in cluster_layers_and_slice_mesh
        raise NotImplementedError("automatically slicing layers with "
    jax._src.traceback_util.UnfilteredStackTrace: NotImplementedError: automatically slicing layers with existing physical meshes is notsupported yet.
    
    The stack trace below excludes JAX-internal frames.
    The preceding is the original exception that occurred, unmodified.
    
    --------------------
    
    The above exception was the direct cause of the following exception:
    
    Traceback (most recent call last):
      File "train.py", line 368, in <module>
        trainer.train()
      File "train.py", line 295, in train
        executable) = compile_and_benchmark_pipeshard_training_executable(
      File "/home/cyxue/Projects/playground/slice_profile/jax/benchmark_parallel_utils.py", line 410, in compile_and_benchmark_pipeshard_training_executable
        latencies, e2e_total_time, niter, local_lats = benchmark_training_executable(
      File "/home/cyxue/Projects/playground/slice_profile/jax/benchmark_parallel_utils.py", line 268, in benchmark_training_executable
        state = train_step(state, batches[i])
      File "/home/cyxue/Projects/playground/alpa/alpa/alpa/pipeline_parallel/compile_executable.py", line 92, in compile_pipeshard_executable
        pipeshard_config = compile_pipeshard_executable_internal(
      File "/home/cyxue/Projects/playground/alpa/alpa/alpa/pipeline_parallel/compile_executable.py", line 177, in compile_pipeshard_executable_internal
        manual_stage_option) = cluster_layers_and_slice_mesh(
      File "/home/cyxue/Projects/playground/alpa/alpa/alpa/pipeline_parallel/stage_construction.py", line 734, in cluster_layers_and_slice_mesh
        raise NotImplementedError("automatically slicing layers with "
    NotImplementedError: automatically slicing layers with existing physical meshes is notsupported yet.
    ```



Wait for your responses and thanks!

