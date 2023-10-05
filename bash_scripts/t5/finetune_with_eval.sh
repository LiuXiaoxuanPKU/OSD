datapath=$1
dataset_name=$2
sample_source=$3
kl=$4
bsz=$5
gradient_step=$6s

python distill/train_mlm.py \
    --student_model_path google/t5-efficient-small  \
    --teacher_model_path google/flan-t5-xl \
    --dataset_name ${dataset_name} \
    --do_train \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${bsz} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${gradient_step} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --source_max_length 1024 \
    --train_target_max_length 512 \
    --val_target_max_length 512 \
    --test_target_max_length 512 \
    --run_name t5_${dataset_name}_online_distill_${sample_source}_${kl} \
    --output_dir $datapath/t5_${dataset_name}_finetune_${sample_source}_${kl}

python distill/finetune_mlm.py \
    --student_model_path $datapath/t5_${dataset_name}_finetune_${sample_source}_${kl}
    --teacher_model_path google/t5-efficient-xl \
    --dataset_name ${dataset_name} \
    --do_predict \
    --per_device_eval_batch_size 1 \
    --logging_steps 1 \
    --source_max_length 1024 \
    --test_target_max_length 512 \
    --run_name eval \
    --output_dir $datapath/t5_${dataset_name}_finetune_evaluation_${sample_source}_${kl}
