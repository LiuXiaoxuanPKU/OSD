datapath=$1
dataset_name=$2
sample_source=$3
kl=$4

python distill/train_mlm.py \
    --student_model_path google/t5-efficient-small  \
    --teacher_model_path google/flan-t5-xl \
    --dataset_name ${dataset_name} \
    --max_propose_num 5 \
    --mode online \
    --sample_source ${sample_source} \
    --all_token_mask True \
    --do_train \
    --num_train_epochs 1 \
    --online_eval_interval 10 \
    --online_update_interval 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --source_max_length 1024 \
    --train_target_max_length 512 \
    --val_target_max_length 512 \
    --test_target_max_length 512 \
    --run_name t5_${dataset_name}_online_distill_${sample_source}_${kl} \
    --output_dir $datapath/t5_${dataset_name}_online_${sample_source}_${kl}
