import numpy as np

BATCH_SIZE = 16
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
    parser.add_argument("--use_stream", type=str2bool, default=True)
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



def inference(args):

    bs = int(args.batch_size)
    inputs_shape = [bs] + list(map(int, args.inputs_shape.split(",")))
    assert len(inputs_shape)==4 and inputs_shape[1] == 3, ""

    input_batch = np.ones(inputs_shape).astype(np.float32)
    
    onnx_file_path = args.onnx_file_path
    engine_file_path = onnx_file_path[:-4] + "trt"
   
    if os.path.exists(engine_file_path):
        os.system(f"rm -rf {engine_file_path}")

    _engine = get_engine(args, onnx_file_path, engine_file_path, inputs_shape)
    del _engine
    # init autolog
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

    f = open(engine_file_path, "rb")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    output = np.empty([bs, 1000], dtype ="float32") 

    # allocate device memory
    d_input = cuda.mem_alloc(1 * input_batch.nbytes)
    d_output = cuda.mem_alloc(1 * output.nbytes)

    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    for i in range(args.warmup_times):
        cuda.memcpy_htod_async(d_input, input_batch, stream)
        # execute model
        context.execute_async_v2(bindings, stream.handle, None)
        # transfer predictions back
        cuda.memcpy_dtoh_async(output, d_output, stream)
        # syncronize threads
        stream.synchronize()
        #print(f"warmup {i} times, the shape of output is {output.shape}")
    print("warmup done")
    
    st = time.time()
    cuda.memcpy_htod(d_input, input_batch)   
    for i in range(args.repeat_times):
        # predict(batch): # result gets copied into output
        # transfer input data to device
        autolog.times.start()
        time.sleep(0.001)
        autolog.times.stamp()
        # execute model
        context.execute_v2(bindings)
        autolog.times.stamp()
        time.sleep(0.001)
        autolog.times.end(stamp=True)

    cuda.memcpy_dtoh(output, d_output)

    print(f"run {args.repeat_times} times done with {time.time() - st}s!")
    # Report log
    autolog.report()


if __name__ == "__main__":
    args = parse_args()
    inference(args)

