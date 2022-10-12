### Note for the Source Code of Alpa

[Link for source code](https://github.com/alpa-projects/alpa/tree/main/alpa)

> NOTE: The following file path description is based on the root path of `./alpa/alpa`, which is the python source code of Alpa.

-------

#### 1. `./device_mesh.py`

- `try_import_ray_worker()`: Imported from `./util.py`

    - Args: error: Whether to raise an error if ray.worker cannot be imported.
    - Tries importing `ray.worker` and returns the module (or None).
    - Returns: The `ray.worker` modules.
    - Raises: ImportError: If error=True and ray's version >= 2.0.

- `list_gpu_info()`: Imported from `./util.py`

    - List all gpu information by calling nvidia-smi.

    ```python
    def list_gpu_info():
        """List all gpu information by calling nvidia-sim."""
        ret = subprocess.getoutput("nvidia-smi -L")
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if visible_devices:
            ids = [int(x) for x in visible_devices.split(",")]
            lines = ret.split("\n")
            lines = [lines[i] for i in ids]
            ret = "\n".join(lines)
        return ret
    ```

- `class MeshHostWorker`:

    - A ray actor that manages the xla computation and buffers on a single host.
    - <font color='red'>The key part to map device mesh into ray devices!</font>
        - **Alpa put a MeshHostWorker Instance on each host, a device mesh can contain several hosts, a device mesh group can contain several meshes.**
    - Maintain buffers
    - Data loader
    - **Cross-mesh Resharding**: 
        - `init_collective_group()`: Initialize the collective group eagerly.
        - `generate_nccl_uid()`: Generate the NCCL unique ID in advance.
        - `init_p2p_communicator()`: Initialize the P2P communicator from within the mesh workers.
        - `init_broadcast_communicator()`: Initialize the broadcast communicator from within the mesh workers.
        - `create_and_set_cross_mesh_communicators()`: Create collective communicators for the cross mesh group.
        - `put_resharding_send_task() / put_resharding_recv_task()`
        - `run_resharding_send_task() / run_resharding_recv_task()`
        - `send_tile() / recv_tile()`
        - `put_resharding_broadcast_task() / run_resharding_broadcast_task()`
    - **Profiling and Debugging**:
        - `profile_hlo_ops()`: call `./mesh_profiling.py`
        - `profile_executable_with_dummy_inputs()`
        - `profile_resharding_send_task() / profile_resharding_recv_task()`: call `run_resharding_send_task() / run_resharding_recv_task()`, run in `benchmark_func()` imported from `./util.py`

- `class PhysicalDeviceMesh(ABC)`:

    - The **base class of physical device mesh**. A physical device mesh is a 2-dimensional mesh that runs SPMD computation on all devices in the mesh.

    - `def get_logical_mesh()`:

        - Return a logical mesh and parameters of the alpha-beta communication cost model. The logical view is used for auto-sharding.

        - Reshape the physical device mesh into the input mesh shape:

            ```python
            id_mesh = np.arange(self.num_devices).reshape(mesh_shape)
            ```

        - Compute bandwidth of doing communication along dim 0 & 1.

        - Return: `Class LogicalDeviceMesh`, defined in `./shard_parallel/auto_sharding.py`

- (TYPE 1) `class LocalPhysicalDeviceMesh(PhysicalDeviceMesh)`:

    - A **single-host physical device mesh to run computation on local devices**. It uses the **native XLA runtime**.

    - `jax.interpreters.pxla`: **JAX toolkit for sharding tensor** from [Link](https://jax.readthedocs.io/en/latest/_modules/jax/interpreters/pxla.html?highlight=jax.interpreters.pxla)

        ```python
        # A ShardingSpec describes at a high level how a logical array is sharded across
        # devices (each ShardedDeviceArray has a ShardingSpec, and ShardingSpecs also
        # describe how to shard inputs to a parallel computation). spec_to_indices()
        # encodes exactly how a given ShardingSpec is translated to device buffers, i.e.
        # how the sharded array is "laid out" across devices. Given a sequence of
        # devices, we shard the data across the devices in row-major order, with
        # replication treated as an extra inner dimension.
        #
        # For example, given the logical data array [1, 2, 3, 4], if we were to
        # partition this array 4 ways with a replication factor of 2, for a total of 8
        # devices, the data on each device would be: [1, 1], [2, 2], [3, 3], [4, 4].
        ```

- (TYPE 2) `class DistributedPhysicalDeviceMesh(PhysicalDeviceMesh)`:

    - A **multi-host physical device mesh to run computation distributedly**. It uses **ray actors** and the **distributed XLA runtime**.

    - **Lanuch distributed XLA runtime**:

        ```python
        # In def _launch_xla_servers(self):
        port = None
        while port in used_port_set:
        port = np.random.randint(20000, 25000)
        used_port_set.add(port)
        
        self.server_address = f"{self.head_ip}:{port}"
        logger.debug(f"Trying to start XLA gRPC server on port: {port}...")
        self.service_server = xla_client._xla.get_distributed_runtime_service(
        self.server_address, self.num_hosts, use_coordination_service=False)
        logger.debug(f"Success to start XLA gRPC server on port: {port}...")
        time.sleep(0.4)
        ```

    -  **Retrieve the placement group**: `retrieve_placement_group()` from `./util.py`

        - Retrieve the placement group to support node affinity scheduling If already inside the placement group, retrieve the current placement group (case I). Then, if the placement group is detected globally in alpa, retrieve the global placement group (case II).

    - **Set XLA environment variables**: 

        ```python
        # Set XLA environment variables
        env_vars = {
          "ALPA_IS_WORKER":
          "True",
          "NCCL_USE_MULTISTREAM":
          "False",
          "XLA_PYTHON_CLIENT_MEM_FRACTION":
          str(global_config.xla_client_mem_fraction),
          "XLA_FLAGS": (os.environ.get("XLA_FLAGS", "") +
                        f" --xla_gpu_autotune_level"
                        f"={global_config.xla_gpu_autotune_level}"),
        
          # "NCCL_LAUNCH_MODE": "PARALLEL",
          # "XLA_FLAGS": "--xla_dump_to=hlo --xla_dump_hlo_pass_re=.*"
          # "NCCL_DEBUG": "INFO" if i == 0 else "VERSION",
          # "NCCL_DEBUG_SUBSYS": "ALL",
          # "RAY_IGNORE_UNHANDLED_ERRORS": "True",
        }
        
        if global_config.resharding_mode == "broadcast":
          env_vars["NCCL_ALGO"] = "Ring"
          env_vars["NCCL_PROTO"] = "Simple"
        
          if "XLA_PYTHON_CLIENT_ALLOCATOR" in os.environ:
            env_vars["XLA_PYTHON_CLIENT_ALLOCATOR"] = os.environ[
              "XLA_PYTHON_CLIENT_ALLOCATOR"]
        
            if "NCCL_DEBUG" in os.environ:
              env_vars["NCCL_DEBUG"] = os.environ[
                "NCCL_DEBUG"] if i == 0 else "VERSION"
        
              if global_config.use_aws_efa:
                env_vars.update({
                  "FI_PROVIDER": "efa",
                  "FI_EFA_USE_DEVICE_RDMA": "1",
                  "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH",
                                                    ""),  # For libnccl-net.so
                })
        ```

    - **Launch worker**:

        ```python
        # Launch the DaemonMoveWorker
        cls = ray.remote(num_cpus=0)(DaemonMoveWorker)
        move_worker = cls.options(
        placement_group=placement_group,
        placement_group_bundle_index=bundle_index).remote()
        
        # Launch the MeshHostWorker
        cls = ray.remote(num_cpus=0,
        								num_gpus=self.num_devices_per_host)(MeshHostWorker)
        worker = cls.options(placement_group=placement_group,
        											placement_group_bundle_index=bundle_index,
        											runtime_env={
                                "env_vars": env_vars
                                }).remote(self.server_address, self.num_hosts,
                                i, self.mesh_id, move_worker,
                                global_config.runtime_random_seed)
        self.workers.append(worker)
        ```

    - `def get_remote_buffers()`: Get values of remote buffers.

        - Args:

            - host_local_ids: For each RemoteArrayRef, we can fetch a list of buffers from multiple devices on multiple hosts. This variable defines a list of (host_id, local_id) pair for each RemoteArrayRef. If it is None, fetch all remote buffers.

            - batching: Whether batch remote calls by host ids. This can reduce ray overhead.

- `class RemoteArrayRef`:

    - A **reference to all device buffers of a logical array**. In Alpa, each pipeshard stage runs in SPMD (single program, multiple device). Hence, buffers of the same logical array are allocated, used and freed together, and thus we use one reference for all these buffers.

- `class DistributedArray`:

    - A **distributed array on a PhysicalDeviceMesh**. End users can interact with this array as if they are working with a normal numpy array. 

        Internally, it **stores a pointer to all remote buffers**. The buffers are stored distributedly on remote workers' device memeory. **When users require the value of the array. These buffers will be gathered to the driver**.

- `class ReplicatedDistributedArray`:

    - A **distributed array that is replicated on multiple meshes**. These class is **used for arrays that need to be replicated on multiple physical meshes (e.g., optimizer's step)**.

- `class VirtualPhysicalMesh`:

    - A **virtual physical mesh used for pipeline parallel compilation**.

        VirtualPhysicalMesh is **used during compile time**. We don't allocate actual workers for it. **When compilation is finished, we instantiated it as a PhysicalDeviceMesh and launch workers**.

        A VirtualPhysicalMesh can also be sliced into multiple VirtualPhysicalMesh. After slicing, each sliced VirtualPhysicalMesh can be instantiated as a PhysicalDeviceMesh. **These sliced PhysicalDeviceMesh together can form a PhysicalDeviceMeshGroup for pipeline parallelism**.

    - `slice_1d()`:

        - Slice a mesh given the slicing config.
        - Args:
            - dim: which dimension to slice from, 0 is host or 1 is the gpu
            - indices: indices to include along this dimension.
        - Returns:
            - mesh (PhysicalDeviceMesh)

    - `slice_2d()`

    - `slice_profiling_submeshes()`: <font color=red>Don't know what for...</font>

- `class PhysicalDeviceMeshGroup`:

    - **A list of physical devices that forms a pipeline.**
    - `establish_nccl_group()`: Establish NCCL group between two meshes.

- `class DeviceCluster`:

    - A **ray cluster with GPU devices**. This is **the top interface for alpa to interact with ray cluster's resources**.

    - `create_placement_group()`: from `./util.py`

        - **Creates a placement group if it does not exist**. If a placement group is already detected (in Tune integration), this will be a no-op.

            **By default the placement group will be created with `SPREAD` strategy**. This is optimized for colocating GPUs on different nodes.

            Args:

            ​        - num_hosts: the number of hosts to create the placement group for

            ​        - host_num_devices: the number of devices on each host

            ​        - additional_resources_per_host: additional resources per host

            Returns:

                    - The placement group

    - `get_physical_mesh()`:

        - **Slice a subset of hosts and devices to form a physical device mesh**.
        - Args:
            - host_ids: The index of host nodes. "None" means using all hosts.
            - num_devices_per_host: The number of devices per host. "None" means using all devices.
        - Return:
            - A physical multi-host device mesh `return DistributedPhysicalDeviceMesh`.

    - `get_virtual_physical_mesh()`

-------



#### 2. `./shard_parallel/auto_sharding.py`

- `class AutoShardingOption`: Options of the auto-sharding solver.

- `class LogicalDeviceMesh`:

    - A **logical view of a physical mesh**. The logical view is **used in the auto-sharding pass**.

    - ***A physical mesh can have multiple logical views**. (e.g., a 2x8 physical mesh can be viewed as a 1x16 or a 4x4 logical mesh). **Each mesh dimension has its own latency and bandwidth**. We use **alpha-beta model to model the communication cost**.

    - `all_gather_cost() / all_reduce_cost() / reduce_scatter_cost() / all_to_all_cost()`

    - **Tensor Sharding & Mesh Mapping**:

        ```python
        def make_tile_spec(self, array, tensor_dims, mesh_dims):
            shape = array.shape
            sharding = [
                pxla.NoSharding(),
            ] * len(shape)
            mesh_mapping = [
                None,
            ] * len(self.id_mesh.shape)
        
            for i, (tensor_dim, mesh_dim) in enumerate(zip(tensor_dims, mesh_dims)):
                sharding[tensor_dim] = pxla.Chunked([self.id_mesh.shape[mesh_dim]],)
                mesh_mapping[mesh_dim] = pxla.ShardedAxis(i)
        
            for i, mapping in enumerate(mesh_mapping):
                if mapping is None:
                    mesh_mapping[i] = pxla.Replicated(self.id_mesh.shape[i])
        
            return pxla.ShardingSpec(sharding, mesh_mapping)
        ```

- `class HloStatus(Enum)`: The status of an HloModule.

    - UNOPTIMIZED, SHARDING_ANNOTATED, SPMD_PARTITIONED, FULLY_OPTIMIZED

- (<font color=red>VITAL</font>) `run_auto_sharding_pass()`:

    - Run the **auto-sharding pass to annotate sharding specs for an XLA Computation**.

    - Args:

        - **hlo_module**: The hlo module **got by tracing the jax function**, whose **status should be UNOPTIMIZED**.
        - logical_mesh: The logical device mesh.
        - return_mode: 
            - **single**:  return **a single HLO module**, whose status is **SPMD_PARTITIONED**;
            - **stages**:  return **HLO modules of multiple pipeline stages**, whose statuses are **SHARDING_ANNOTATED**.
            - **stages_and_hook**: return **HLO modules of multiple pipeline stages and the hooked hlo sharding**. The statuses of the returned protos are **SHARDING_ANNOTATED**.

        - num_micro_batches: **The number of micro batches if gradient accumulation is used**. If this is set, the cost of all-reduce for gradient synchronization is divided by this number.
        - memory_budget_per_device: The memory budget per device in bytes.
        - `as_option: AutoShardingOption`: **Determine the tensor sharding spec option**.

    - `get_compile_options()`: import from `./util.py`

        - Specify **num_replicas, num_partitions, device_assignment, use_spmd_partitioning** and so on. **Return CompileOptions for XLA compilation**.

    - `class XlaPassContext`: import from `./util.py`

        - A **global context for passing arguments from python to XLA c++ passes**.

    - `with XlaPassContext(...): xe.run_auto_sharding(hlo_module, compile_options)`

        - **Run auto-sharding**.

    - `class StagePlan`: import from `./parallel_plan.py`

        - **The parallel plan for a single sharded stage.**

- (<font color=red>VITAL</font>) `run_spmd_partitioner_pass()`:

    - **Run SPMD partitioner pass on a sharding annotated HLO Module for obtaining pipeline STAGES PARTITION ** (no need for the hlo module in single mode).
    - Args: 
        - **hlo_module**: The **input HLO module**, whose status should be **SHARDING_ANNOTATED**.
        - num_devices: The total number of devices.
    - `get_compile_options()`
    - `with XlaPassContext(...): xe.run_spmd_partitioner(hlo_module, compile_options)`
        - **Run SPMD partitioner**, partition **based on sharding annotation in `hlo_module`**.

- (<font color=red>VITAL</font>) `run_backend_compilation()`:

    - **Compile a spmd partitioned Hlo Module to an XLA executable**.
    - Args:
        - **backend: The XLA backend client**.
        - **hlo_module**: The **input HLO Module**, whose status should be **SPMD_PARTITIONED**.
        - **stage_plan**: The **auto-sharding strategy solution**. **Note that still need to specify the parallel plan for single sharded stage!**
        - num_devices: The total number of devices.
    - `with XlaPassContext(...): compiled = backend.compile(xla_computation, compile_options)`

- (<font color=red>VITAL</font>) `get_input_output_sharding_specs()`:

    - **Get the sharding specs of input/output tensors from a HloModule**, which means **resharding the input/output tensors based on sharding spec of OPs**. 
    - Args:
        - **hlo_module: The sharded HLO module**.
        - **avals: The abstract values of input tensors**.
        - **out_avals: The abstract values of output tensors**.
        - num_devices: The total number of devices.
        - **logical_mesh_shape: The shape of logical mesh**.
    - Returns:
        - **input_sharding_specs: The sharding specs of input tensors**.
        - **output_sharding_specs: The sharding specs of output tensors**.

- `hlo_sharding_to_sharding_spec()`:

    - **Convert hlo sharding to sharding spec**.

- (<font color=red>VITAL</font>) `_call_solver_serialized_args()`:

    - **Call the solver with serialized arguments, optimize for the intra-op paralleliem plan for stage-mesh pair**. 

- `set_auto_sharded_hlo_stages()`:
    - Set the sliced auto-sharded stages. This is **called in XLA SliceAutoShardedStages pass (in tensorflow-alpa)**.
- `get_auto_sharded_hlo_stages()`: **Get the sliced hlo stages from the SliceAutoShardedStages pass**.

-------



#### 3. `./mesh_executable.py`

A **mesh executable** encapsulates **all compiled binary and meta information of a distributed executable**.

A mesh executable **contains one or several XLA executables**. For each type of mesh executable, there is a **driver part** and a **worker par**t. The **driver part runs on the user script** and the **worker parts run on distributed workers**. The **driver part sends control commands to launch the worker parts on workers**.

- `class MeshDriverExecutable(ABC)`: 
    - **The base class of the driver part of a mesh executable.**
    - `launch_on_driver()`: Launch the executable on the driver.
    - `get_parallel_plan()`: Get the overall parallel plan.
    - `profile_with_dummy_inputs()`: Profile the execution time costs with dummy inputs.
    - `get_execution_time_costs()`: Return the pure execution time costs recorded by an internal timer.
    - `get_shard_args_time_costs()`: Return the time costs of sharding input arguments.
    - `get_hlo_text()`: Return the HLO IR in the text format.
    - `get_total_allocation_size()`: Get the total memory allocation size in bytes.
    - `sync()`: Sync all workers.

- `class MeshWorkerExecutable(ABC)`:

    - **The base class of the worker part of a mesh executable.**
    - `execute_on_worker()`: Run the executable on the worker.
    - `profile_with_dummy_inputs()`: Profile the execution time costs with dummy inputs.
    - `get_hlo_text()`: Return the HLO IR in the text format.
    - `get_total_allocation_size()`: Get the total memory allocation size in bytes.

- `get_sync_func_driver(physical_mesh)`: 

    - Get the sync function on the driver by calling `physical_mesh.devices[0].synchronize_all_activity()`.

- `get_sync_func_worker(worker)`:

    - Get the sync function on the workers by calling `worker.local_devices[0].synchronize_all_activity()`.

- `class NormalMeshDriverExecutable(MeshDriverExecutable)`:

    - **The driver part of a normal mesh executable.**
    - `_set_executable(self, physical_mesh, hlo_module, stage_plan)`:
        - **Put the executable on workers**.
            - If is **distributed physical device mesh**: `MeshHostWorker.put_executable.remote(...)` for each worker in the mesh;
            - If is **local physical device mesh** (only a worker on the same host with this driver, **no need to dispatch executables**): `self.compiled = run_backend_compilation(...)`. 

    - `launch_on_driver()`: **Launch the executable on the driver**.

        - If is **distributed physical device mesh**:

            - **Execute the SPMD binary**: 

                ```python
                # Execute the SPMD binary
                for i in range(num_hosts):
                		physical_mesh.workers[i].run_executable.remote(
                				self.exec_uuid, input_uuids, output_uuids, **kwargs)
                ```

            - **Gather output buffers**:

                ```python
                # Gather output buffers
                output_bufs = np.array(
                					[RemoteArrayRef(physical_mesh, uuid) for uuid in output_uuids])
                ```

        - If is **local physical device mesh**:

            - **Connect the input buffers to the self.compiled and gather output buffers**:

                ```python
                output_bufs = self.compiled.execute_sharded_on_local_devices(input_bufs)
                ```

    - `__call__()`: Fast call without signature matching:  `out = self.launch_on_driver(*args_flat)`

    - `profile_with_dummy_inputs()`: **Call Ray**

        ```python
        def profile_with_dummy_inputs(self, **kwargs):
            """Profile the execution time costs with dummy inputs."""
            if isinstance(self.physical_mesh, DistributedPhysicalDeviceMesh):
                tasks = []
                for worker in self.physical_mesh.workers:
                    tasks.append(
                        worker.profile_executable_with_dummy_inputs.remote(
                        self.exec_uuid, **kwargs))
                costs = ray.get(tasks)
                for cost_vec in costs:
                    if np.inf in cost_vec:
                        return [np.inf] * len(cost_vec)
                costs = np.mean(costs, axis=0)
            else:
                assert isinstance(self.physical_mesh, LocalPhysicalDeviceMesh)
                costs = profile_xla_executable(self.compiled,
                    self.physical_mesh.backend,
                    self.physical_mesh.devices)
            return costs
        ```

        - `profile_xla_executable()`: import from `./util.py`

            - **Measure the time costs of a xla executable with dummy inputs**.

            ```python
            # Run benchmark
            def run_func():
            		device_outputs = compiled.execute_sharded_on_local_devices(device_inputs)
            ```

- `class NormalMeshWorkerExecutable(MeshWorkerExecutable)`:

    - **The worker part of a normal mesh executable. Seems only for distributed scenarios**. 

    - (In init) `self.compiled = run_backend_compilation(worker.backend, hlo_proto, stage_plan, num_devices)`

    - `execute_on_worker()`: 

        - **Run the executable on the worker. Only for distributed scenarios since in local scenarios, executable is started running by _set_executable() in the driver**.

        ```py
        # Execute the executable
        timers(self.timer_name).start(self.sync_func if sync_before else None)
        try:
        		output_bufs = self.compiled.execute_sharded_on_local_devices(
        				input_bufs)
        except RuntimeError:
        		ray.actor.exit_actor()
        timers(self.timer_name).stop(self.sync_func if sync_after else None)
        
        # Store output buffers
        for i in range(len(output_uuids)):
        		buffer_dict[output_uuids[i]] = output_bufs[i]
        ```

    - `profile_with_dummy_inputs()`: 

        ```python
        return profile_xla_executable(self.compiled, backend, local_devices)
        ```

- `get_grad_sync_channel_ids(hlo_module: xe.HloModule)`:

    - **Return the channel ids of all-reduces that are used for gradient synchronization**.

- `class GradAccMeshDriverExecutable(MeshDriverExecutable)`:

    - **The driver part of a gradient accumulation mesh executable.**

- `class GradAccMeshWorkerExecutable(MeshWorkerExecutable)`:

    - **The worker part of a gradient accumulation mesh executable.**

- `PartialGradAccMeshDriverExecutable(NormalMeshDriverExecutable)`:

    - **The driver part of a mesh executable that can optionally skip the gradient synchronization step.**
    - **This executable is used for computation stages in pipeline, such as forward, backward and apply_grad**.
