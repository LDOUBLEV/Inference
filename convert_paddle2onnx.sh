INFERENCE_MODEL_DIR="./inference"
MODEL_LIST=${INFERENCE_MODEL_DIR}/model_list.txt

for model in `cat ${MODEL_LIST}`;do
    paddle2onnx --model_dir ./${INFERENCE_MODEL_DIR}/${model} \
                --model_filename ./${INFERENCE_MODEL_DIR}/${model}/inference.pdmodel \
                --params_filename ./${INFERENCE_MODEL_DIR}/${model}/inference.pdiparams \
                --save_file ./onnx_model_opset11/${model}.onnx \
                --opset_version 11 \
                --enable_onnx_checker True
done
