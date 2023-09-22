datapath=$1

WANDB_PROJECT=specInfer python distill/train.py \
    --student_model_path $datapath/online_sample \
    --teacher_model_path $datapath/vicuna-7b-v1.3/ \
    --data_path data/sharp.json \
    --max_propose_num 5 \
    --bf16 True \
    --output_dir $datapath/online_sample \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0. \
    --lr_scheduler_type "constant" \
    --tf32 True \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --run_name sharp_online_baseline \
    --mode online \
    --online_eval_interval 1 \
    --online_update_interval 1000000 \
    --logging_steps 1 \
    --logging_nan_inf_filter true 