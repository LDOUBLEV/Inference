import numpy as np
import argparse
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import auto_log

import onnx
import onnxruntime as rt


import time

def str2bool(v):
    return v.lower() in ("true", "t", "1")


def init_args():
    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_thread", type=int, default=1)
    parser.add_argument("--repeat_times", type=int, default=1000)
    parser.add_argument("--warmup_times", type=int, default=100)
    parser.add_argument("--inputs_shape", type=str, default="3,224,224")
    parser.add_argument("--onnx_file_path", type=str, default="./onnx_model_opset11/GhostNet_x0_5.onnx")
    
    return parser


def parse_args():
    parser = init_args()
    return parser.parse_args()


def build_sess(onnx_file_path="", num_thread=1):
    so = rt.SessionOptions()
    so.intra_op_num_threads = num_thread
    so.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
     
    onnx_model = onnx.load_model(onnx_file_path)
    sess = rt.InferenceSession(onnx_model.SerializeToString(), sess_options=so)

    return sess

    

def inference(args):

    bs = int(args.batch_size)
    inputs_shape = [bs] + list(map(int, args.inputs_shape.split(",")))
    assert len(inputs_shape)==4 and inputs_shape[1] == 3, ""

    fake_inputs = np.ones(inputs_shape).astype(np.float32)
    onnx_file_path = args.onnx_file_path

    onnx_name = os.path.basename(args.onnx_file_path)

    pid = os.getpid()
    autolog = auto_log.AutoLogger(
                model_name=onnx_name,
                model_precision="fp32",
                batch_size=args.batch_size,
                data_shape=inputs_shape,
                save_path="./output/infer.log",
                pids=pid,
                process_name=None,
                inference_config={"cpu_math_library_num_threads": args.num_thread},
                gpu_ids=0,
                time_keys=[
                    'preprocess_time', 'inference_time', 'postprocess_time'
                ],
                warmup=0)

    # Do inference
    print('Running inference on image fake_image ...'.format())

    st = time.time()
    sess = build_sess(onnx_file_path=onnx_file_path, num_thread=args.num_thread)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    for i in range(args.warmup_times):
        output = sess.run([output_name], {input_name : fake_inputs})
        prob = np.squeeze(output[0])
        #run()

    print(f"run {args.warmup_times} times warmup done with {time.time() - st}s!")

    st = time.time()
    for i in range(args.repeat_times):
        autolog.times.start()
        # Preprocess
        autolog.times.stamp()
        # Run inference.
        output = sess.run([output_name], {input_name : fake_inputs})
        autolog.times.stamp()
        # PostPreprocess.
        prob = np.squeeze(output[0])
        autolog.times.end(stamp=True)
    print(f"run {args.repeat_times} times done with {time.time() - st}s!")
    # Report log
    autolog.report()


if __name__ == "__main__":
    args = parse_args()
    inference(args)

