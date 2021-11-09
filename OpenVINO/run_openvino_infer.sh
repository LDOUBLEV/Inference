tag=
niter=1000
nwarmup=100
batchsize_list="1 16"
nthreads_list="1 6"
filename=$1
models_dir=$2
models=`cat ${filename}`


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


mkdir output
for nthreads in ${nthreads_list}
do
    for batchsize in ${batchsize_list}
    do
        for model in $models
        do
            model_path=${models_dir}/${model}/${model}.xml
            if [ -d $model_path ]
            then
                echo "The model no exists:"${model}
                continue
            fi
            sleep 10s
            python3 -u ./inference_openvino.py -m ${model_path} --niter ${niter} --nwarmup ${nwarmup} --nthreads ${nthreads} --batch_size ${batchsize} --save_path ./autolog/${model}_bs_${batchsize}_threads_${nthreads}.log &> ./output/${model}_bs_${batchsize}_threads_${nthreads}.log
            status_check $? "${cmd}" "${status_log}"
            echo "Done: $model."
        done
    done
done
