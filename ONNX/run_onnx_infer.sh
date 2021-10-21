#!/bin/bash 

FILENAME=$1
dataline=$(cat ${FILENAME})

ONNX_MODEL_DIR="../../onnx_model_opset11"
IFS=$'\n'
lines=(${dataline})
batch_size=( 1 16 )
NUM_THREADS=( 1 6 )
USE_GPU=( True False )


eval "mkdir output"
status_log="./output/results.log"


function status_check(){
    last_status=$1   # the exit code
    run_command=$2
    run_log=$3
    if [ $last_status -eq 0 ]; then
        echo -e "\033[33m Run successfully with command - ${run_command}!  \033[0m" | tee -a ${run_log}
    else
        echo -e "\033[33m Run failed with command - ${run_command}!  \033[0m" | tee -a ${run_log}
    fi
}

for onnx_name in ${lines[*]}; do
    onnx_path="${ONNX_MODEL_DIR}/${onnx_name}"
    for bs in ${batch_size[*]}; do
        for threads in ${NUM_THREADS[*]}; do
            for use_gpu in ${USE_GPU[*]}; do
                _save_log_path="./output/onnx_${onnx_name}_precision_fp32_batchsize_${bs}_threads_${threads}_usegpu_${use_gpu}.log"
                cmd="python3.7 inference_onnx.py --onnx_file_path=${onnx_path}  --batch_size=${bs} --num_thread=${threads} --use_gpu=${use_gpu}> ${_save_log_path} 2>&1 "
                eval ${cmd}
                echo ${cmd}
                status_check $? "${cmd}" "${status_log}"
                sleep 0.1
            done
        done
    done
done

cmd="python3.7 ../utils/log_parser.py --log_path=./output/ --output_name=ONNX_onnx_fp32.excel.xlsx"
eval $cmd


