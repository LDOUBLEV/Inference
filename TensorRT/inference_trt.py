import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common
import auto_log

TRT_LOGGER = trt.Logger()

import time

def str2bool(v):
    return v.lower() in ("true", "t", "1")


def init_args():
    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--repeat_times", type=int, default=1000)
    parser.add_argument("--warmup_times", type=int, default=100)
    parser.add_argument("--inputs_shape", type=str, default="3,224,224")
    parser.add_argument("--onnx_file_path", type=str, default="")

    return parser


def parse_args():
    parser = init_args()
    return parser.parse_args()


def get_engine(args, onnx_file_path, engine_file_path="", input_shape=[1,3,224,224]):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 28 # 256MiB
            builder.max_batch_size = args.batch_size
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = input_shape
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


class FakeInferConfig:
    batch_size = 1
    inputs_shape = [3, 224, 224]
    repeat_times = 1000
    warmup_times = 100
    onnx_file_path = "/paddle/package/TensorRT-7.2.3.4/samples/python/clas_onnx/onnx_model_opset11/MobileNetV2.onnx"
    

def inference(args):

    bs = int(args.batch_size)
    inputs_shape = [bs] + list(map(int, args.inputs_shape.split(",")))
    assert len(inputs_shape)==4 and inputs_shape[1] == 3, ""

    fake_inputs = np.ones(inputs_shape).astype(np.float32)
    onnx_file_path = args.onnx_file_path

    engine_file_path = onnx_file_path[:-4] + "trt"

    if os.path.exists(engine_file_path):
        os.system(f"rm -rf {engine_file_path}")

    onnx_name = os.path.basename(args.onnx_file_path)

    pid = os.getpid()
    autolog = auto_log.AutoLogger(
                model_name=onnx_name,
                model_precision="fp32",
                batch_size=args.batch_size,
                data_shape=inputs_shape,
                save_path="./output/infer.log",
                inference_config=None,
                pids=pid,
                process_name=None,
                gpu_ids=0,
                time_keys=[
                    'preprocess_time', 'inference_time', 'postprocess_time'
                ],
                warmup=0)

    with get_engine(args, onnx_file_path, engine_file_path, inputs_shape) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        print('Running inference on image fake_image ...'.format())
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = fake_inputs
        st = time.time()
        for i in range(args.warmup_times):
            # trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            # This function is generalized for multiple inputs/outputs.
            # inputs and outputs are expected to be lists of HostDeviceMem objects.
            # Transfer input data to the GPU.
            [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
            # Run inference.
            context.execute_async(batch_size=args.batch_size, bindings=bindings, stream_handle=stream.handle)
            # Transfer predictions back from the GPU.
            [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
            # Synchronize the stream
            stream.synchronize()
            # Return only the host outputs.
            trt_outputs = [out.host for out in outputs]
        
        print(f"run {args.warmup_times} times warmup done with {time.time() - st}s!")

        st = time.time()
        output_shapes = [(1, 1000)]
        for i in range(args.repeat_times):
            autolog.times.start()
            # Preprocess 
            [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
            autolog.times.stamp()
            # Run inference.
            context.execute_async(batch_size=args.batch_size, bindings=bindings, stream_handle=stream.handle)
            autolog.times.stamp()
            # Transfer predictions back from the GPU.
            [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
            # Synchronize the stream
            stream.synchronize()
            # Return only the host outputs.
            trt_outputs = [out.host for out in outputs]
            trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
            autolog.times.end(stamp=True)
        print(f"run {args.repeat_times} times done with {time.time() - st}s!")
        # Report log
        autolog.report()


if __name__ == "__main__":
    args = parse_args()
    # args.onnx_file_path = "/paddle/package/TensorRT-7.2.3.4/samples/python/clas_onnx/onnx_model_opset11/MobileNetV2_x0_75.onnx"
    inference(args)

