#!/bin/bash 

FILENAME=$1
dataline=$(cat ${FILENAME})

IFS=$'\n'
lines=(${dataline})
batch_size=( 1 8 16 32 )

out_dir="output_TRT_synch"
eval "mkdir ${out_dir}"
status_log="./${out_dir}/results.log"


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
    onnx_path="./onnx_model_opset11/${onnx_name}"
    for bs in ${batch_size[*]}; do
        _save_log_path="./${out_dir}/trt_${onnx_name}_precision_fp32_batchsize_${bs}.log"
        cmd="python3.7 inference_trt_synch.py --onnx_file_path=${onnx_path}  --batch_size=${bs} --use_stream=False > ${_save_log_path} 2>&1 "
        eval $cmd
        status_check $? "${cmd}" "${status_log}"
        sleep 0.1
    done
done

cmd="python3.7 ../utils/log_parser.py --log_path=./${out_dir}/ --output_name=./${out_dir}/TRT_onnx_fp32_synch.excel.xlsx"
eval $cmd


