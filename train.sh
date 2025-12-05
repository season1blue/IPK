export CUDA_VISIBLE_DEVICES=0
datasets=(GQA)
model_names=(4_origin)
for dataset in ${datasets[@]}
do
    for model_name in ${model_names[@]}
    do
        python train.py \
            --llm_model 7B \
            --llama_model_path /ai/teacher/dkc/Assets/origin/weights/ \
            --max_seq_len 512 \
            --batch_size 8 \
            --accum_iter 1 \
            --epochs 1 \
            --warmup_epochs 2 \
            --blr 9e-3 \
            --weight_decay 0.02 \
            --output_dir ./ckpts/\
            --log_dir ./logs/\
            --tsbd ./tsbd/\
            --adapter_dim 12 \
            --adapter_scale 0.1 \
            --prompt_format QCM-A \
            --seed 0 \
            --emb 320 \
            --dataset ${dataset} \
            --model_name ${model_name}
    done
done