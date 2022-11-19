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

------



All right, it seems that the 'profile worker forcely killed' is not related to batch size, as I also get the same error when I try to profile with batch size = 16...

------



@zhisbug Thx, any advisement on sloving the **'NotImplementedError'** can be really helpful for me. 

Actually **I've found the reason that why the 'profile worker forcely killed' ('failed to start the RPC server') error occurred**. **To fix this bug, I need to make a small change to `def lanuch_xla_servers()` function in `device_mesh.py`**.

The process of finding this bug is as follows:

I've added some logged information in the `device_mesh.py/class DistributedPhysicalDeviceMesh/def launch_xla_servers()` like:

```python
def launch_xla_servers(self):
  print("[I] Lanuching distributed XLA servers...")

  # Launch distributed xla runtime
  port = None
  while port in used_port_set:
    port = np.random.randint(20000, 25000)
    used_port_set.add(port)

    print("[I] The selected port is {} while the used port is as follows:".format(port))
    print("    -", used_port_set)


    ######################################################
    # NOTE: EXPERIMENTAL

    # NOTE: Get ray node ip address
    @ray.remote
    def f():
      time.sleep(0.01)
      return ray._private.services.get_node_ip_address()

    ip_addr = ray.get(f.remote())

    print("[I] Current IP address:", ip_addr)

    # Server address
    # server_address = f"{self.head_ip}:{port}"
    server_address = f"{ip_addr}:{port}"

    ######################################################

    print("[I] The target server address is: {}".format(server_address))

    logger.debug(f"Trying to start XLA gRPC server on port: {port}...")

    print("[I] Querying XLA API client to get XLA gRPC server...")

    service_server = xla_client._xla.get_distributed_runtime_service(
      server_address, self.num_hosts, use_coordination_service=False)

    print("[I] XLA gRPC server is started successfully.")

    logger.debug(f"Success to start XLA gRPC server on port: {port}...")
    time.sleep(0.4)

    # Worker launch is not shown here...
```

And after I run the distributed training among two nodes mentioned in my first comment, I got the following log information:

```bash
(ProfileWorker pid=1119457, ip=10.2.64.52) [I] Lanuching distributed XLA servers...
(ProfileWorker pid=1119457, ip=10.2.64.52) [I] The selected port is 24142 while the used port is as follows:
(ProfileWorker pid=1119457, ip=10.2.64.52)     - {24142, None}
(ProfileWorker pid=1119457, ip=10.2.64.52) E1115 18:17:20.306562478 1119457 server_chttp2.cc:40]        {"created":"@1668507440.306478244","description":"No address added out of total 1 resolved","file":"external/com_github_grpc_grpc/src/core/ext/transport/chttp2/server/chttp2_server.cc","file_line":395,"referenced_errors":[{"created":"@1668507440.306471548","description":"Unable to configure socket","fd":44,"file":"external/com_github_grpc_grpc/src/core/lib/iomgr/tcp_server_utils_posix_common.cc","file_line":216,"referenced_errors":[{"created":"@1668507440.306456068","description":"Cannot assign requested address","errno":99,"file":"external/com_github_grpc_grpc/src/core/lib/iomgr/tcp_server_utils_posix_common.cc","file_line":189,"os_error":"Cannot assign requested address","syscall":"bind"}]}]}
WARNING:alpa.pipeline_parallel.stage_profiling:Meet unexpected error, all profile workers are forcely killed
(ProfileWorker pid=1119457, ip=10.2.64.52) 2022-11-15 18:17:20,316	ERROR worker.py:451 -- Exception raised in creation task: The actor died because of an error raised in its creation task, ray::ProfileWorker.__init__() (pid=1119457, ip=10.2.64.52, repr=<alpa.pipeline_parallel.stage_profiling.ProfileWorker object at 0x7f538fcd39a0>)

(ProfileWorker pid=1119457, ip=10.2.64.52)   File "/home/cyxue/Projects/playground/alpa/alpa/alpa/pipeline_parallel/stage_profiling.py", line 251, in __init__


(ProfileWorker pid=1119457, ip=10.2.64.52)     self.mesh = virtual_mesh.get_physical_mesh()
- Profiling for submesh 2 (1, 4):
(ProfileWorker pid=1119457, ip=10.2.64.52)   File "/home/cyxue/Projects/playground/alpa/alpa/alpa/device_mesh.py", line 1934, in get_physical_mesh
- Generate all stage infos (Jaxpr -> HLO)
(ProfileWorker pid=1119457, ip=10.2.64.52)     self.launched_physical_mesh = DistributedPhysicalDeviceMesh(
(ProfileWorker pid=1119457, ip=10.2.64.52)   File "/home/cyxue/Projects/playground/alpa/alpa/alpa/device_mesh.py", line 974, in __init__
  0%|                                                                                                             | 0/2 [00:00<?, ?it/s]
(ProfileWorker pid=1119457, ip=10.2.64.52)     self.service_server, self.workers = self.launch_xla_servers()
(ProfileWorker pid=1119457, ip=10.2.64.52)   File "/home/cyxue/Projects/playground/alpa/alpa/alpa/device_mesh.py", line 1030, in launch_xla_servers

(ProfileWorker pid=1119457, ip=10.2.64.52)     service_server = xla_client._xla.get_distributed_runtime_service(
  0%|                                                                                                             | 0/2 [00:00<?, ?it/s]
(ProfileWorker pid=1119457, ip=10.2.64.52) [I] Current IP address: 10.2.64.52or: UNKNOWN: Failed to start RPC server
(ProfileWorker pid=1119457, ip=10.2.64.52) [I] The target server address is: 10.2.64.51:24142
(ProfileWorker pid=1119457, ip=10.2.64.52) [I] Querying XLA API client to get XLA gRPC server...
```

I found that when lanuching distributed XLA server, the `10.2.64.52` (worker node) had tried to start the RPC server on `10.2.64.51:24142`, where `10.2.64.51` is the ip addr for head node and `24142` is the random port obtained by the worker node. Obviously, the worker node cannot access this port on the head node (even may not exposed). 

After tracing back to the `self.head_ip`, I found that you were trying to initialize this ip addr with:

```py
ray_global_node = ray_worker._global_node
try:
  self.head_info = ray_global_node.address_info
except AttributeError as ae:
  raise RuntimeError(
    "Cannot access ray global node. Did you call ray.init?") \
  from ae
  self.head_ip = self.head_info["node_ip_address"]
```

However, this method will always return the ip addr of the head node. Therefore, when the worker node executes the `lanuch_xla_servers()` function, it will resort to the wrong ip addr to start a RPC server.

As a correction, I have made a small change in `lanuch_xla_servers()` (also given in the first code piece):

```python
    ######################################################
    # NOTE: EXPERIMENTAL

    # NOTE: Get ray node ip address
    @ray.remote
    def f():
      time.sleep(0.01)
      return ray._private.services.get_node_ip_address()

    ip_addr = ray.get(f.remote())

    print("[I] Current IP address:", ip_addr)

    # Server address, comment the original code
    # server_address = f"{self.head_ip}:{port}"
    # And change to
    server_address = f"{ip_addr}:{port}"

    ######################################################
```

In this way, the worker node can also get the correct ip addr and all things goes well after this change.

```bash
(ProfileWorker pid=1121827, ip=10.2.64.52) [I] Lanuching distributed XLA servers...
(ProfileWorker pid=1121827, ip=10.2.64.52) [I] The selected port is 24149 while the used port is as follows:
(ProfileWorker pid=1121827, ip=10.2.64.52)     - {24149, None}
(ProfileWorker pid=2608095) [I] Hosts num: 1 | Devices num: 4
(ProfileWorker pid=1121827, ip=10.2.64.52) [I] Current IP address: 10.2.64.52
(ProfileWorker pid=1121827, ip=10.2.64.52) [I] The target server address is: 10.2.64.52:24149
(ProfileWorker pid=1121827, ip=10.2.64.52) [I] Querying XLA API client to get XLA gRPC server...
(ProfileWorker pid=1121827, ip=10.2.64.52) [I] XLA gRPC server is started successfully.
(ProfileWorker pid=1121827, ip=10.2.64.52) [I] Hosts num: 1 | Devices num: 4
```

