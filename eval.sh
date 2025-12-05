export CUDA_VISIBLE_DEVICES=0
datasets=(VQAv2 VisWiz)
model_names=(0_ours 1_zipped 2_full 3_hierachical 4_origin)
for dataset in ${datasets[@]}
do
    for model_name in ${model_names[@]}
    do
        python eval.py \
        --dataset ${dataset} \
        --model_name ${model_name}
    done
done