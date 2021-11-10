Help on ICudaEngine in module tensorrt.tensorrt object:

class ICudaEngine(pybind11_builtins.pybind11_object)
 |  An :class:`ICudaEngine` for executing inference on a built network.
 |
 |  The engine can be indexed with ``[]`` . When indexed in this way with an integer, it will return the corresponding binding name. When indexed with a string, it will return the corresponding binding index.
 |
 |  :ivar num_bindings: :class:`int` The number of binding indices.
 |  :ivar max_batch_size: :class:`int` The maximum batch size which can be used for inference. For an engine built from an :class:`INetworkDefinition` without an implicit batch dimension, this will always be ``1`` .
 |  :ivar has_implicit_batch_dimension: :class:`bool` Whether the engine was built with an implicit batch dimension.. This is an engine-wide property. Either all tensors in the engine have an implicit batch dimension or none of them do. This is True if and only if the :class:`INetworkDefinition` from which this engine was built was created with the ``NetworkDefinitionCreationFlag.EXPLICIT_BATCH`` flag.
 |  :ivar num_layers: :class:`int` The number of layers in the network. The number of layers in the network is not necessarily the number in the original :class:`INetworkDefinition`, as layers may be combined or eliminated as the :class:`ICudaEngine` is optimized. This value can be useful when building per-layer tables, such as when aggregating profiling data over a number of executions.
 |  :ivar max_workspace_size: :class:`int` The amount of workspace the :class:`ICudaEngine` uses. The workspace size will be no greater than the value provided to the :class:`Builder` when the :class:`ICudaEngine` was built, and will typically be smaller. Workspace will be allocated for each :class:`IExecutionContext` .
 |  :ivar device_memory_size: :class:`int` The amount of device memory required by an :class:`IExecutionContext` .
 |  :ivar refittable: :class:`bool` Whether the engine can be refit.
 |  :ivar name: :class:`str` The name of the network associated with the engine. The name is set during network creation and is retrieved after building or deserialization.
 |  :ivar num_optimization_profiles: :class:`int` The number of optimization profiles defined for this engine. This is always at least 1.
 |
 |  Method resolution order:
 |      ICudaEngine
 |      pybind11_builtins.pybind11_object
 |      builtins.object
 |
 |  Methods defined here:
 |
 |  __del__(...)
 |      __del__(self: tensorrt.tensorrt.ICudaEngine) -> None
 |
 |  __enter__ = common_enter(this)
 |      # Provides Python's `with` syntax
 |
 |  __exit__ = common_exit(this, exc_type, exc_value, traceback)
 |      Destroy this object, freeing all memory associated with it. This should be called to ensure that the object is cleaned up properly.
 |      Equivalent to invoking :func:`__del__`
 |
 |  __getitem__(...)
 |      __getitem__(*args, **kwargs)
 |      Overloaded function.
 |
 |      1. __getitem__(self: tensorrt.tensorrt.ICudaEngine, arg0: str) -> int
 |
 |      2. __getitem__(self: tensorrt.tensorrt.ICudaEngine, arg0: int) -> str
 |
 |  __init__(self, /, *args, **kwargs)
 |      Initialize self.  See help(type(self)) for accurate signature.
 |
 |  __len__(...)
 |      __len__(self: tensorrt.tensorrt.ICudaEngine) -> int
 |
 |
 |  binding_is_input(...)
 |      binding_is_input(*args, **kwargs)
 |      Overloaded function.
 |
 |      1. binding_is_input(self: tensorrt.tensorrt.ICudaEngine, index: int) -> bool
 |
 |
 |                  Determine whether a binding is an input binding.
 |
 |                  :index: The binding index.
 |
 |                  :returns: True if the index corresponds to an input binding and the index is in range.
 |
 |
 |      2. binding_is_input(self: tensorrt.tensorrt.ICudaEngine, name: str) -> bool
 |
 |
 |                  Determine whether a binding is an input binding.
 |
 |                  :name: The name of the tensor corresponding to an engine binding.
 |
 |                  :returns: True if the index corresponds to an input binding and the index is in range.
 |
 |  create_execution_context(...)
 |      create_execution_context(self: tensorrt.tensorrt.ICudaEngine) -> tensorrt.tensorrt.IExecutionContext
 |
 |
 |      Create an :class:`IExecutionContext` .
 |
 |      :returns: The newly created :class:`IExecutionContext` .
 |
 |  create_execution_context_without_device_memory(...)
 |      create_execution_context_without_device_memory(self: tensorrt.tensorrt.ICudaEngine) -> tensorrt.tensorrt.IExecutionContext
 |
 |
 |      Create an :class:`IExecutionContext` without any device memory allocated
 |      The memory for execution of this device context must be supplied by the application.
 |
 |      :returns: An :class:`IExecutionContext` without device memory allocated.
 |
 |  get_binding_bytes_per_component(...)
 |      get_binding_bytes_per_component(self: tensorrt.tensorrt.ICudaEngine, index: int) -> int
 |
 |
 |      Return the number of bytes per component of an element.
 |      The vector component size is returned if :func:`get_binding_vectorized_dim` != -1.
 |
 |      :arg index: The binding index.
 |
 |  get_binding_components_per_element(...)
 |      get_binding_components_per_element(self: tensorrt.tensorrt.ICudaEngine, index: int) -> int
 |
 |
 |      Return the number of components included in one element.
 |
 |      The number of elements in the vectors is returned if :func:`get_binding_vectorized_dim` != -1.
 |
 |      :arg index: The binding index.
 |
 |  get_binding_dtype(...)
 |      get_binding_dtype(*args, **kwargs)
 |      Overloaded function.
 |
 |      1. get_binding_dtype(self: tensorrt.tensorrt.ICudaEngine, index: int) -> tensorrt.tensorrt.DataType
 |
 |
 |                  Determine the required data type for a buffer from its binding index.
 |
 |                  :index: The binding index.
 |
 |                  :Returns: The type of data in the buffer.
 |
 |
 |      2. get_binding_dtype(self: tensorrt.tensorrt.ICudaEngine, name: str) -> tensorrt.tensorrt.DataType
 |
 |
 |                  Determine the required data type for a buffer from its binding index.
 |
 |                  :name: The name of the tensor corresponding to an engine binding.
 |
 |                  :Returns: The type of data in the buffer.
 |
 |  get_binding_format(...)
 |      get_binding_format(self: tensorrt.tensorrt.ICudaEngine, index: int) -> tensorrt.tensorrt.TensorFormat
 |
 |
 |      Return the binding format.
 |
 |      :arg index: The binding index.
 |
 |  get_binding_format_desc(...)
 |      get_binding_format_desc(self: tensorrt.tensorrt.ICudaEngine, index: int) -> str
 |
 |
 |      Return the human readable description of the tensor format.
 |
 |      The description includes the order, vectorization, data type, strides, etc. For example:
 |
 |      |   Example 1: kCHW + FP32
 |      |       "Row major linear FP32 format"
 |      |   Example 2: kCHW2 + FP16
 |      |       "Two wide channel vectorized row major FP16 format"
 |      |   Example 3: kHWC8 + FP16 + Line Stride = 32
 |      |       "Channel major FP16 format where C % 8 == 0 and H Stride % 32 == 0"
 |
 |      :arg index: The binding index.
 |
 |  get_binding_index(...)
 |      get_binding_index(self: tensorrt.tensorrt.ICudaEngine, name: str) -> int
 |
 |
 |      Retrieve the binding index for a named tensor.
 |
 |
 |      You can also use engine's :func:`__getitem__` with ``engine[name]``. When invoked with a :class:`str` , this will return the co
rresponding binding index.
 |
 |      :func:`IExecutionContext.execute_async()` and :func:`IExecutionContext.execute()` require an array of buffers.
 |      Engine bindings map from tensor names to indices in this array.
 |      Binding indices are assigned at :class:`ICudaEngine` build time, and take values in the range [0 ... n-1] where n is the total
number of inputs and outputs.
 |
 |      :arg name: The tensor name.
 |
 |      :returns: The binding index for the named tensor, or -1 if the name is not found.
 |
 |  get_binding_name(...)
 |      get_binding_name(self: tensorrt.tensorrt.ICudaEngine, index: int) -> str
 |
 |
 |      Retrieve the name corresponding to a binding index.
 |
 |      You can also use engine's :func:`__getitem__` with ``engine[index]``. When invoked with an :class:`int` , this will return the
corresponding binding name.
 |
 |      This is the reverse mapping to that provided by :func:`get_binding_index()` .
 |
 |      :arg index: The binding index.
 |
 |      :returns: The name corresponding to the binding index.
 |
 |  get_binding_shape(...)
 |      get_binding_shape(*args, **kwargs)
 |      Overloaded function.
 |
 |      1. get_binding_shape(self: tensorrt.tensorrt.ICudaEngine, index: int) -> tensorrt.tensorrt.Dims
 |
 |
 |                  Get the shape of a binding.
 |
 |                  :index: The binding index.
 |
 |                  :Returns: The shape of the binding if the index is in range, otherwise Dims()
 |
 |
 |      2. get_binding_shape(self: tensorrt.tensorrt.ICudaEngine, name: str) -> tensorrt.tensorrt.Dims
 |
 |
 |                  Get the shape of a binding.
 |
 |                  :name: The name of the tensor corresponding to an engine binding.
 |
 |                  :Returns: The shape of the binding if the tensor is present, otherwise Dims()
 |
 |  get_binding_vectorized_dim(...)
 |      get_binding_vectorized_dim(self: tensorrt.tensorrt.ICudaEngine, index: int) -> int
 |
 |
 |      Return the dimension index that the buffer is vectorized.
 |
 |      Specifically -1 is returned if scalars per vector is 1.
 |
 |      :arg index: The binding index.
 |
 |  get_location(...)
 |      get_location(*args, **kwargs)
 |      Overloaded function.
 |
 |      1. get_location(self: tensorrt.tensorrt.ICudaEngine, index: int) -> tensorrt.tensorrt.TensorLocation
 |
 |
 |                  Get location of binding.
 |                  This lets you know whether the binding should be a pointer to device or host memory.
 |
 |                  :index: The binding index.
 |
 |                  :returns: The location of the bound tensor with given index.
 |
 |
 |      2. get_location(self: tensorrt.tensorrt.ICudaEngine, name: str) -> tensorrt.tensorrt.TensorLocation
 |
 |
 |                  Get location of binding.
 |                  This lets you know whether the binding should be a pointer to device or host memory.
 |
 |                  :name: The name of the tensor corresponding to an engine binding.
 |
 |                  :returns: The location of the bound tensor with given index.
 |
 |  get_profile_shape(...)
 |      get_profile_shape(*args, **kwargs)
 |      Overloaded function.
 |
 |      1. get_profile_shape(self: tensorrt.tensorrt.ICudaEngine, profile_index: int, binding: int) -> List[tensorrt.tensorrt.Dims]
 |
 |
 |                  Get the minimum/optimum/maximum dimensions for a particular binding under an optimization profile.
 |
 |                  :arg profile_index: The index of the profile.
 |                  :arg binding: The binding index or name.
 |
 |                  :returns: A ``List[Dims]`` of length 3, containing the minimum, optimum, and maximum shapes, in that order.
 |
 |
 |      2. get_profile_shape(self: tensorrt.tensorrt.ICudaEngine, profile_index: int, binding: str) -> List[tensorrt.tensorrt.Dims]
 |
 |
 |                  Get the minimum/optimum/maximum dimensions for a particular binding under an optimization profile.
 |
 |                  :arg profile_index: The index of the profile.
 |                  :arg binding: The binding index or name.
 |
 |                  :returns: A ``List[Dims]`` of length 3, containing the minimum, optimum, and maximum shapes, in that order.
 |
 |  get_profile_shape_input(...)
 |      get_profile_shape_input(*args, **kwargs)
 |      Overloaded function.
 |
 |
 |      1. get_profile_shape_input(self: tensorrt.tensorrt.ICudaEngine, profile_index: int, binding: int) -> List[List[int]]
 |
 |
 |                  Get minimum/optimum/maximum values for an input shape binding under an optimization profile. If the specified bindi
ng is not an input shape binding, an exception is raised.
 |
 |                  :arg profile_index: The index of the profile.
 |                  :arg binding: The binding index or name.
 |
 |                  :returns: A ``List[List[int]]`` of length 3, containing the minimum, optimum, and maximum values, in that order. If
 the values have not been set yet, an empty list is returned.
 |
 |
 |      2. get_profile_shape_input(self: tensorrt.tensorrt.ICudaEngine, profile_index: int, binding: str) -> List[List[int]]
 |
 |
 |                  Get minimum/optimum/maximum values for an input shape binding under an optimization profile. If the specified bindi
ng is not an input shape binding, an exception is raised.
 |
 |                  :arg profile_index: The index of the profile.
 |                  :arg binding: The binding index or name.
 |
 |                  :returns: A ``List[List[int]]`` of length 3, containing the minimum, optimum, and maximum values, in that order. If
 the values have not been set yet, an empty list is returned.
 |
 |  is_execution_binding(...)
 |      is_execution_binding(self: tensorrt.tensorrt.ICudaEngine, binding: int) -> bool
 |
 |
 |      Returns :class:`True` if tensor is required for execution phase, false otherwise.
 |
 |      For example, if a network uses an input tensor with binding i ONLY as the reshape dimensions for an :class:`IShuffleLayer` , th
en ``is_execution_binding(i) == False``, and a binding of `0` can be supplied for it when calling :func:`IExecutionContext.execute` or
:func:`IExecutionContext.execute_async` .
 |
 |      :arg binding: The binding index.
 |
 |  is_shape_binding(...)
 |      is_shape_binding(self: tensorrt.tensorrt.ICudaEngine, binding: int) -> bool
 |
 |
 |      Returns :class:`True` if tensor is required as input for shape calculations or output from them.
 |
 |      TensorRT evaluates a network in two phases:
 |
 |      1. Compute shape information required to determine memory allocation requirements and validate that runtime sizes make sense.
 |
 |      2. Process tensors on the device.
 |
 |      Some tensors are required in phase 1. These tensors are called "shape tensors", and always
 |      have type :class:`tensorrt.int32` and no more than one dimension. These tensors are not always shapes
 |      themselves, but might be used to calculate tensor shapes for phase 2.
 |
 |      :func:`is_shape_binding` returns true if the tensor is a required input or an output computed in phase 1.
 |      :func:`is_execution_binding` returns true if the tensor is a required input or an output computed in phase 2.
 |
 |      For example, if a network uses an input tensor with binding ``i`` as an input to an IElementWiseLayer that computes the reshape
 dimensions for an :class:`IShuffleLayer` , ``is_shape_binding(i) == True``
 |
 |      It's possible to have a tensor be required by both phases. For instance, a tensor can be used as a shape in an :class:`IShuffle
Layer` and as the indices for an :class:`IGatherLayer` collecting floating-point data.
 |
 |      It's also possible to have a tensor required by neither phase that shows up in the engine's inputs. For example, if an input te
nsor is used only as an input to an :class:`IShapeLayer` , only its shape matters and its values are irrelevant.
 |
 |      :arg binding: The binding index.
 |
 |  serialize(...)
 |      serialize(self: tensorrt.tensorrt.ICudaEngine) -> tensorrt.tensorrt.IHostMemory
 |
 |
 |      Serialize the engine to a stream.
 |
 |      :returns: An :class:`IHostMemory` object containing the serialized :class:`ICudaEngine` .
 |
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |
 |  device_memory_size
 |
 |  has_implicit_batch_dimension
 |
 |  max_batch_size
 |
 |  max_workspace_size
 |
 |  name
 |
 |  num_bindings
 |
 |  num_layers
 |
 |  num_optimization_profiles
 |
 |  refittable
 |
 |  ----------------------------------------------------------------------
 |  Static methods inherited from pybind11_builtins.pybind11_object:
 |
 |  __new__(*args, **kwargs) from pybind11_builtins.pybind11_type
 |      Create and return a new object.  See help(type) for accurate signature.



