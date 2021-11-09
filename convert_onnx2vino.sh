filename=$1
models_dir=$2
input_shape=[1,3,224,224]
models=`cat ${filename}`

for model in $models
do
    python3 /opt/intel/openvino_2021/deployment_tools/model_optimizer/mo.py --input_model ${models_dir}/${model}.onnx --data_type FP32 --output_dir ./openvino_model/${model}/ --input_shape ${input_shape}
done
