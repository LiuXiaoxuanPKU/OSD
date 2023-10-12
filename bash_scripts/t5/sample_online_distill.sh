datapath=$1
dataset_name=$2
sample_source=$3
kl=$4

python distill/train_mlm.py \
    --student_model_path google/t5-efficient-small \
    --teacher_model_path google/flan-t5-xl \
    --dataset_name ${dataset_name} \
    --max_propose_num 5 \
    --output_dir $datapath/${dataset_name}_online_distill_${sample_source}_${kl} \
    --do_train \
    --num_train_epochs 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 4 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --source_max_length 1024 \
    --test_target_max_length 512 \
    --run_name t5_${dataset_name}_online_distill_${sample_source}_${kl} \
    --mode offline \
    --sample_source ${sample_source} \
    --kl_method ${kl} \
    --report_to none

python3 distill/train_mlm.py \
    --student_model_path $datapath/${dataset_name}_online_distill_${sample_source}_${kl} \
    --teacher_model_path google/flan-t5-xl \
    --dataset_name ${dataset_name} \
    --max_propose_num 5 \
    --output_dir $datapath/${dataset_name}_online_baseline_${sample_source}_${kl} \
    --do_train \
    --fast_eval \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0. \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --source_max_length 1024 \
    --test_target_max_length 512 \
    --run_name ${dataset_name}_online_baseline_${sample_source}_${kl} \
    --mode online \
    --sample_source ${sample_source} \
    --kl_method ${kl} \
    --online_eval_interval 1 \
    --online_update_interval 100000 \
    --logging_steps 1 \
    --logging_nan_inf_filter true