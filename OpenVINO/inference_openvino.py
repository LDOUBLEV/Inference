"""
This code implements is borrowed from https://github.com/openvinotoolkit/openvino
"""

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import numpy as np
from openvino.inference_engine import IECore

import auto_log


class NormalizeImage(object):
    def __init__(self, scale=None, mean=None, std=None):
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, img):
        return (img.astype('float32') * self.scale - self.mean) / self.std


def build_argparser():
    parser = ArgumentParser()
    args = parser.add_argument_group('Options')
    args.add_argument("-m", "--model", help="Required. Path to an .xml or .onnx file with a trained model.",
                      required=True, type=str)
    args.add_argument("-d", "--device", default="CPU", type=str)
    # nthreads must be str type
    args.add_argument("--nthreads", default="1", type=str)
    #TODO(gaotingquan): support GPU
    args.add_argument("--gpu_id", default=0, type=int)
    args.add_argument("--nireq", '--num_infer_requests', help='Optional. Number of infer requests', default=1, type=int)
    args.add_argument("--batch_size", default=1, type=int)
    args.add_argument("--niter", default=1000, type=int)
    args.add_argument("--nwarmup", default=100, type=int)
    args.add_argument("--save_path", default=None, type=str)
    args.add_argument("--autolog", default=True, type=bool)
    return parser


def main():
    args = build_argparser().parse_args()

    #TODO(gaotingquan): support GPU
    if args.device.upper() != "CPU":
        msg = "Only support CPU device."
        raise Exception(msg)

    # Plugin initialization for specified device and load extensions library if specified
    ie = IECore()
    ie.set_config({"CPU_THREADS_NUM" : args.nthreads}, "CPU")
    ie.set_config({"CPU_BIND_THREAD": "YES"}, "CPU")

    # Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
    model = args.model
    net = ie.read_network(model=model)

    assert len(net.input_info.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    net.batch_size = args.batch_size
    n, c, h, w = net.input_info[input_blob].input_data.shape

    # Loading model to the plugin
    exec_net = ie.load_network(network=net, device_name=args.device)
    request_id = 0
    infer_request = exec_net.requests[request_id]

    # build input image
    images = []
    for i in range(args.batch_size):
        image = np.random.randint(0, 255, size=(h, w, c))
        images.append(image)

    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    img_scale = 1.0 / 255.0
    normalize_op = NormalizeImage(scale=img_scale, mean=img_mean, std=img_std)

    # Autolog
    if args.autolog:
        model_name=args.model.split("/")[-2]
        pid = os.getpid()
        autolog = auto_log.AutoLogger(
            model_name=model_name,
            model_precision="FP32",
            batch_size=args.batch_size,
            data_shape=f"{c}*{h}*{w}",
            save_path=args.save_path,
            inference_config={"cpu_math_library_num_threads": args.nthreads},
            pids=pid,
            process_name=None,
            gpu_ids=args.gpu_id if args.device.upper() == "GPU" else None,
            time_keys=['preprocess_time', 'inference_time', 'postprocess_time'],
            warmup=args.nwarmup,
            run_devices=f"{args.device}",
            ir_optim=False,
            enable_tensorrt=False,
            enable_mkldnn=False,
            cpu_threads=0,
            enable_mem_optim=False)

    if args.nwarmup:
        image = np.random.rand(c, h, w).astype(np.float32)
        inputs = [image * args.batch_size]
        for i in range(args.nwarmup):
            infer_request.infer({input_blob: inputs})

    for i in range(args.niter):
        if args.autolog:
            autolog.times.start()
        inputs = []
        for image in images:
            image = image[:, :, ::-1]
            image = normalize_op(image)
            image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            inputs.append(image)
        if args.autolog:
            autolog.times.stamp()

        infer_request.infer({input_blob: inputs})
        if args.autolog:
            autolog.times.stamp()

        res = infer_request.output_blobs[out_blob]
        if args.autolog:
            autolog.times.end(stamp=True)

    if args.autolog:
        autolog.report()


if __name__ == '__main__':
    sys.exit(main() or 0)
