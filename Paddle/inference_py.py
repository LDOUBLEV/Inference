import os
import argparse
import time
import numpy as np
import paddle.inference as paddle_infer
import auto_log

def str2bool(v):
    return v.lower() in ("true", "t", "1")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="the dir of model")
    parser.add_argument("--warmup_times", type=int, default=100, help="warmup")
    parser.add_argument("--repeats_times", type=int, default=1000, help="repeats")
    parser.add_argument("--threads", type=int, default=1, help="threads")
    parser.add_argument("--enable_mkldnn", type=str2bool, help="enable mkldnn")
    parser.add_argument("--enable_gpu", type=str2bool, default="true", help="enable gpu")
    parser.add_argument("--enable_trt", type=str2bool, default="true", help="enable trt")
    parser.add_argument("--enable_profile", type=str2bool, default="False", help="enable profile")
    parser.add_argument("--batch_size", type=int, default=1, help="batch_size")
    parser.add_argument("--input_shape", type=str, default="3,224,224", help="enable profile")
    args = parser.parse_args()
    assert (args.model_dir is not None) or \
            ((args.model_file is not None) and (args.params_file is not None)), \
            "Input error. If the model is uncombined, set model_dir." \
            "Otherwise, set model_file and params_file."
    return parser.parse_args()

def set_config(args):
    if os.path.exists(args.model_dir):
        model_file = os.path.join(args.model_dir, "inference.pdmodel")
        model_params = os.path.join(args.model_dir, "inference.pdiparams")
        config = paddle_infer.Config(model_file, model_params)
    else:
        raise ValueError(f"The model dir {args.model_dir} does not exists!")
    
    # enable memory optim
    config.enable_memory_optim()
    #config.disable_gpu()
    config.set_cpu_math_library_num_threads(args.threads)
    config.switch_ir_optim(True)
    if args.enable_mkldnn and not args.enable_gpu:
        config.disable_gpu()
        config.enable_mkldnn()
    if args.enable_profile:
        config.enable_profile()
    if args.enable_gpu:
        config.enable_use_gpu(256, 0)
        if args.enable_trt:
            config.enable_tensorrt_engine(
                precision_mode=paddle_infer.PrecisionType.Float32,
                max_batch_size=20,
                min_subgraph_size=3)
    #config.disable_glog_info()
    pass_builder = config.pass_builder()
    #pass_builder.append_pass('interpolate_mkldnn_pass')
    return config

def main():
    args = parse_args()
    config = set_config(args)
    # set all inputs
    input_shape = args.input_shape
    _ins_shape = [args.batch_size] + list(map(int, input_shape.split(',')))

    model_name = args.model_dir.split("/")[-1]
    # init auto_log  
    pid = os.getpid()
    autolog = auto_log.AutoLogger(
                model_name=model_name,
                model_precision="fp32",
                batch_size=args.batch_size,
                data_shape=_ins_shape,
                save_path="./output/infer.log",
                inference_config=config,
                pids=pid,
                process_name=None,
                gpu_ids=0,
                time_keys=[
                    'preprocess_time', 'inference_time', 'postprocess_time'
                ],
                warmup=0)


    predictor = paddle_infer.create_predictor(config)
    
    fake_input = np.ones(_ins_shape, dtype=np.float32)
    # get input tensor
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])
    input_handle.reshape(_ins_shape)
    # get output tensor
    output_names = predictor.get_output_names()
    output_tensors = []
    for output_name in output_names:
        output_tensor = predictor.get_output_handle(output_name)
        output_tensors.append(output_tensor)

    input_handle.copy_from_cpu(fake_input)
    # run predictor
    for i in range(args.warmup_times):
        input_handle.copy_from_cpu(fake_input)
        predictor.run()

        outputs = []
        for output_tensor in output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)
    print(f"warmup {args.warmup_times} times done!")
    start_time = time.time()
    for i in range(args.repeats_times):
        autolog.times.start()
        input_handle.copy_from_cpu(fake_input)

        autolog.times.stamp()
        predictor.run()

        autolog.times.stamp()
        outputs = []
        for output_tensor in output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)
        autolog.times.end(stamp=True)

    end_time = time.time()
    print(f"repeat {args.repeats_times} times done with {end_time - start_time}s!")
    autolog.report()

if __name__ == "__main__":
    main()

