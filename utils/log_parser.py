import os
import sys
import re
import argparse
 
import pandas as pd
 
def parse_args():
    """
    parse input args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="./log",
                        help="benchmark log path")
    parser.add_argument("--output_name", type=str, default="benchmark_excel.xlsx",
                        help="output excel file name")
    parser.add_argument("--process_trt", dest="process_trt", action='store_true')
    return parser.parse_args()
 
 
def find_all_logs(path_walk : str):
    """
    find all .log files from target dir
    """
    for root, ds, files in os.walk(path_walk):
        for file_name in files:
            if re.match(r'.*.log', file_name):
                full_path = os.path.join(root, file_name)
                yield file_name, full_path
 
def process_log(file_name : str) -> dict:
    """
    process log to dict
    """
    output_dict = {}
    with open(file_name, 'r') as f:
        for i, data in enumerate(f.readlines()):
            if i == 0:
                continue
            line_lists = data.split(" ")
 
            # conf info
            if "runtime_device:" in line_lists:
                pos_buf = line_lists.index("runtime_device:")
                output_dict["runtime_device"] = line_lists[pos_buf + 1].strip()
            if "ir_optim:" in line_lists:
                pos_buf = line_lists.index("ir_optim:")
                output_dict["ir_optim"] = line_lists[pos_buf + 1].strip()
            if "enable_memory_optim:" in line_lists:
                pos_buf = line_lists.index("enable_memory_optim:")
                output_dict["enable_memory_optim"] = line_lists[pos_buf + 1].strip()
            if "enable_tensorrt:" in line_lists:
                pos_buf = line_lists.index("enable_tensorrt:")
                output_dict["enable_tensorrt"] = line_lists[pos_buf + 1].strip()
            if "precision:" in line_lists:
                pos_buf = line_lists.index("precision:")
                output_dict["precision"] = line_lists[pos_buf + 1].strip()
            if "enable_mkldnn:" in line_lists:
                pos_buf = line_lists.index("enable_mkldnn:")
                output_dict["enable_mkldnn"] = line_lists[pos_buf + 1].strip()
            if "cpu_math_library_num_threads:" in line_lists:
                pos_buf = line_lists.index("cpu_math_library_num_threads:")
                output_dict["cpu_math_library_num_threads"] = line_lists[pos_buf + 1].strip()
 
            # model info
            if "model_name:" in line_lists:
                pos_buf = line_lists.index("model_name:")
                output_dict["model_name"] = list(filter(None, line_lists[pos_buf + 1].strip().split('/')))[-1]
            
            # data info
            if "batch_size:" in line_lists:
                pos_buf = line_lists.index("batch_size:")
                output_dict["batch_size"] = line_lists[pos_buf + 1].strip()
            if "input_shape:" in line_lists:
                pos_buf = line_lists.index("input_shape:")
                output_dict["input_shape"] = line_lists[pos_buf + 1].strip()
            
            # perf info
            if "cpu_rss(MB):" in line_lists:
                pos_buf = line_lists.index("cpu_rss(MB):")
                output_dict["cpu_rss(MB)"] = line_lists[pos_buf + 1].strip()
            if "gpu_rss(MB):" in line_lists:
                pos_buf = line_lists.index("gpu_rss(MB):")
                output_dict["gpu_rss(MB)"] = line_lists[pos_buf + 1].strip()
            if "gpu_util:" in line_lists:
                pos_buf = line_lists.index("gpu_util:")
                output_dict["gpu_util"] = line_lists[pos_buf + 1].strip()
            if "preprocess_time(ms):" in line_lists:
                pos_buf = line_lists.index("preprocess_time(ms):")
                output_dict["preprocess_time(ms)"] = line_lists[pos_buf + 1].strip()
            if "inference_time(ms):" in line_lists:
                pos_buf = line_lists.index("inference_time(ms):")
                output_dict["inference_time(ms)"] = line_lists[pos_buf + 1].strip()
            if "postprocess_time(ms):" in line_lists:
                pos_buf = line_lists.index("postprocess_time(ms):")
                output_dict["postprocess_time(ms)"] = line_lists[pos_buf + 1].strip()
        #print(output_dict)
        #sys.exit(0)
    return output_dict
 
 
def main():
    """
    main
    """
    args = parse_args()
    # create empty DataFrame
    origin_df = pd.DataFrame(columns=["model_name",
                                      "batch_size",
                                      "input_shape",
                                      "runtime_device",
                                      "ir_optim",
                                      "enable_memory_optim",
                                      "enable_tensorrt",
                                      "precision",
                                      "enable_mkldnn",
                                      "cpu_math_library_num_threads",
                                      "preprocess_time(ms)",
                                      "inference_time(ms)",
                                      "postprocess_time(ms)",
                                      "cpu_rss(MB)",
                                      "gpu_rss(MB)",
                                      "gpu_util"])
 
    for file_name, full_path in find_all_logs(args.log_path):
        print(file_name, full_path)
        dict_log = process_log(full_path)
        origin_df = origin_df.append(dict_log, ignore_index=True)
 
    raw_df = origin_df.sort_values(by='model_name')
    raw_df.to_excel(args.output_name)
    print(raw_df)
 
if __name__ == "__main__":
    main()
