Help on IExecutionContext in module tensorrt.tensorrt object:

class IExecutionContext(pybind11_builtins.pybind11_object)
 |  Context for executing inference using an :class:`ICudaEngine` .
 |  Multiple :class:`IExecutionContext` s may exist for one :class:`ICudaEngine` instance, allowing the same
 |  :class:`ICudaEngine` to be used for the execution of multiple batches simultaneously.
 |
 |  :ivar debug_sync: :class:`bool` The debug sync flag. If this flag is set to true, the :class:`ICudaEngine` will log the successful execution for each kernel during execute(). It has no effect when using execute_async().
 |  :ivar profiler: :class:`IProfiler` The profiler in use by this :class:`IExecutionContext` .
 |  :ivar engine: :class:`ICudaEngine` The associated :class:`ICudaEngine` .
 |  :ivar name: :class:`str` The name of the :class:`IExecutionContext` .
 |  :ivar device_memory: :class:`capsule` The device memory for use by this execution context. The memory must be aligned on a 256-byte boundary, and its size must be at least :attr:`engine.device_memory_size`. If using :func:`execute_async` to run the network, The memory is in use from the invocation of :func:`execute_async` until network execution is complete. If using :func:`execute`, it is in use until :func:`execute` returns. Releasing or otherwise using the memory for other purposes during this time will result in undefined behavior.
 |  :ivar active_optimization_profile: :class:`int` The active optimization profile for the context. The selected profile will be used in subsequent calls to :func:`execute` or :func:`execute_async` . Profile 0 is selected by default. Changing this value will invalidate all dynamic bindings for the current execution context, so that they have to be set again using :func:`set_binding_shape` before calling either :func:`execute` or :func:`execute_async` .
 |  :ivar all_binding_shapes_specified: :class:`bool` Whether all dynamic dimensions of input tensors have been specified by calling :func:`set_binding_shape` . Trivially true if network has no dynamically shaped input tensors.
 |  :ivar all_shape_inputs_specified: :class:`bool` Whether values for all input shape tensors have been specified by calling :func:`set_shape_input` . Trivially true if network has no input shape bindings.
 |
 |  Method resolution order:
 |      IExecutionContext
 |      pybind11_builtins.pybind11_object
 |      builtins.object
 |
 |  Methods defined here:
 |
 |  __del__(...)
 |      __del__(self: tensorrt.tensorrt.IExecutionContext) -> None
 |
 |  __enter__ = common_enter(this)
 |      # Provides Python's `with` syntax
 |
 |  __exit__ = common_exit(this, exc_type, exc_value, traceback)
 |      Destroy this object, freeing all memory associated with it. This should be called to ensure that the object is cleaned up properly.
 |      Equivalent to invoking :func:`__del__`
 |
 |  __init__(self, /, *args, **kwargs)
 |      Initialize self.  See help(type(self)) for accurate signature.
 |
 |  execute(...)
 |      execute(self: tensorrt.tensorrt.IExecutionContext, batch_size: int = 1, bindings: List[int]) -> bool
 |
 |
 |      Synchronously execute inference on a batch.
 |      This method requires a array of input and output buffers. The mapping from tensor names to indices can be queried using :func:`ICudaEngine.get_binding_index()` .
 |
 |      :arg batch_size: The batch size. This is at most the value supplied when the :class:`ICudaEngine` was built.
 |      :arg bindings: A list of integers representing input and output buffer addresses for the network.
 |
 |      :returns: True if execution succeeded.
 |
 |  execute_async(...)
 |      execute_async(self: tensorrt.tensorrt.IExecutionContext, batch_size: int = 1, bindings: List[int], stream_handle: int, input_co
nsumed: capsule = None) -> bool
 |
 |
 |      Asynchronously execute inference on a batch.
 |      This method requires a array of input and output buffers. The mapping from tensor names to indices can be queried using :func:`
ICudaEngine::get_binding_index()` .
 |
 |      :arg batch_size: The batch size. This is at most the value supplied when the :class:`ICudaEngine` was built.
 |      :arg bindings: A list of integers representing input and output buffer addresses for the network.
 |      :arg stream_handle: A handle for a CUDA stream on which the inference kernels will be executed.
 |      :arg input_consumed: An optional event which will be signaled when the input buffers can be refilled with new data
 |
 |      :returns: True if the kernels were executed successfully.
 |
 |  execute_async_v2(...)
 |      execute_async_v2(self: tensorrt.tensorrt.IExecutionContext, bindings: List[int], stream_handle: int, input_consumed: capsule =
None) -> bool
 |
 |
 |      Asynchronously execute inference on a batch.
 |      This method requires a array of input and output buffers. The mapping from tensor names to indices can be queried using :func:`
ICudaEngine::get_binding_index()` .
 |      This method only works for execution contexts built from networks with no implicit batch dimension.
 |
 |      :arg bindings: A list of integers representing input and output buffer addresses for the network.
 |      :arg stream_handle: A handle for a CUDA stream on which the inference kernels will be executed.
 |      :arg input_consumed: An optional event which will be signaled when the input buffers can be refilled with new data
 |
 |      :returns: True if the kernels were executed successfully.
 |
 |  execute_v2(...)
 |      execute_v2(self: tensorrt.tensorrt.IExecutionContext, bindings: List[int]) -> bool
 |
 |
 |      Synchronously execute inference on a batch.
 |      This method requires a array of input and output buffers. The mapping from tensor names to indices can be queried using :func:`
ICudaEngine.get_binding_index()` .
 |      This method only works for execution contexts built from networks with no implicit batch dimension.
 |
 |      :arg bindings: A list of integers representing input and output buffer addresses for the network.
 |
 |      :returns: True if execution succeeded.
 |
 |  get_binding_shape(...)
 |      get_binding_shape(self: tensorrt.tensorrt.IExecutionContext, binding: int) -> tensorrt.tensorrt.Dims
 |
 |
 |      Get the dynamic shape of a binding.
 |
 |      If :func:`set_binding_shape` has been called on this binding (or if there are no
 |      dynamic dimensions), all dimensions will be positive. Otherwise, it is necessary to
 |      call :func:`set_binding_shape` before :func:`execute_async` or :func:`execute` may be called.
 |
 |      If the ``binding`` is out of range, an invalid Dims with nbDims == -1 is returned.
 |
 |      If ``ICudaEngine.binding_is_input(binding)`` is :class:`False` , then both
 |      :attr:`all_binding_shapes_specified` and :attr:`all_shape_inputs_specified` must be :class:`True`
 |      before calling this method.
 |
 |      :arg binding: The binding index.
 |
 |      :returns: A :class:`Dims` object representing the currently selected shape.
 |
 |  get_shape(...)
 |      get_shape(self: tensorrt.tensorrt.IExecutionContext, binding: int) -> List[int]
 |
 |
 |      Get values of an input shape tensor required for shape calculations or an output tensor produced by shape calculations.
 |
 |      :arg binding: The binding index of an input tensor for which ``ICudaEngine.is_shape_binding(binding)`` is true.
 |
 |      If ``ICudaEngine.binding_is_input(binding) == False``, then both
 |      :attr:`all_binding_shapes_specified` and :attr:`all_shape_inputs_specified` must be :class:`True`
 |      before calling this method.
 |
 |      :returns: An iterable containing the values of the shape tensor.
 |
 |  get_strides(...)
 |      get_strides(self: tensorrt.tensorrt.IExecutionContext, binding: int) -> tensorrt.tensorrt.Dims
 |
 |
 |      Return the strides of the buffer for the given binding.
 |
 |      Note that strides can be different for different execution contexts with dynamic shapes.
 |
 |      :arg binding: The binding index.
 |
 |  set_binding_shape(...)
 |      set_binding_shape(self: tensorrt.tensorrt.IExecutionContext, binding: int, shape: tensorrt.tensorrt.Dims) -> bool
 |
 |
 |      Set the dynamic shape of a binding.
 |
 |      Requires the engine to be built without an implicit batch dimension.
 |      The binding must be an input tensor, and all dimensions must be compatible with
 |      the network definition (i.e. only the wildcard dimension -1 can be replaced with a
 |      new dimension > 0). Furthermore, the dimensions must be in the valid range for the
 |      currently selected optimization profile.
 |
 |      For all dynamic non-output bindings (which have at least one wildcard dimension of -1),
 |      this method needs to be called after setting :attr:`active_optimization_profile` before
 |      either :func:`execute_async` or :func:`execute` may be called. When all input shapes have been
 |      specified, :attr:`all_binding_shapes_specified` is set to :class:`True` .
 |
 |      :arg binding: The binding index.
 |      :arg shape: The shape to set.
 |
 |      :returns: :class:`False` if an error occurs (e.g. index out of range), else :class:`True` .
 |
 |  set_optimization_profile_async(...)
 |      set_optimization_profile_async(self: tensorrt.tensorrt.IExecutionContext, profile_index: int, stream_handle: int) -> bool
 |
 |
 |      Set the optimization profile with async semantics
 |
 |      :arg profile_index: The index of the optimization profile
 |
 |      :arg stream_handle: cuda stream on which the work to switch optimization profile can be enqueued
 |
 |      When an optimization profile is switched via this API, TensorRT may require that data is copied via cudaMemcpyAsync. It is the
 |      application’s responsibility to guarantee that synchronization between the profile sync stream and the enqueue stream occurs.
 |
 |      :returns: :class:`True` if the optimization profile was set successfully
 |
 |  set_shape_input(...)
 |      set_shape_input(self: tensorrt.tensorrt.IExecutionContext, binding: int, shape: List[int]) -> bool
 |
 |
 |      Set values of an input shape tensor required by shape calculations.
 |
 |      :arg binding: The binding index of an input tensor for which ``ICudaEngine.is_shape_binding(binding)`` and ``ICudaEngine.bindin
g_is_input(binding)`` are both true.
 |      :arg shape: An iterable containing the values of the input shape tensor. The number of values should be the product of the dime
nsions returned by ``get_binding_shape(binding)``.
 |
 |      If ``ICudaEngine.is_shape_binding(binding)`` and ``ICudaEngine.binding_is_input(binding)`` are both true, this method must be c
alled before :func:`execute_async` or :func:`execute` may be called. Additionally, this method must not be called if either ``ICudaEngi
ne.is_shape_binding(binding)`` or ``ICudaEngine.binding_is_input(binding)`` are false.
 |
 |      :returns: :class:`True` if the values were set successfully.
 |
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |
 |  active_optimization_profile
 |
 |  all_binding_shapes_specified
 |
 |  all_shape_inputs_specified
 |
 |  debug_sync
 |
 |  device_memory
 |
 |  engine
 |
 |  name
 |
 |  profiler
 |
 |  ----------------------------------------------------------------------
 |  Static methods inherited from pybind11_builtins.pybind11_object:
 |
 |  __new__(*args, **kwargs) from pybind11_builtins.pybind11_type
 |      Create and return a new object.  See help(type) for accurate signature.

