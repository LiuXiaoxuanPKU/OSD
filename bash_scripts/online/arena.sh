datapath=$1
datafile=$2

WANDB_PROJECT=spec python distill/train.py \
    --student_model_path $datapath/llama-160m \
    --teacher_model_path $datapath/vicuna-7b-v1.3/ \
    --data_path arena/$datafile.json \
    --max_propose_num 5 \
    --bf16 True \
    --output_dir $datapath/arena_$datafile \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0. \
    --lr_scheduler_type "constant" \
    --tf32 True \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --run_name arena_${datafile} \
    --mode online \
    --online_eval_interval 1 \
    --online_update_interval 1 \
    --logging_steps 1 \
    --logging_nan_inf_filter true
