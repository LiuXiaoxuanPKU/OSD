WANDB_PROJECT=specInfer python distill/train.py \
    --student_model_path /rscratch/zhendong/lily/llama-160m \
    --teacher_model_path /rscratch/zhendong/lily/vicuna-7b-v1.3/ \
    --data_path data/piqa_train.json \
    --max_propose_num 5 \
    --bf16 True \
    --output_dir /rscratch/zhendong/lily/llama160m_piqa_online_interval1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --run_name llama160m_piqa_online_interval1 \
    --mode online \
    --online_eval_interval 10 \
    --online_update_interval 1 \
    --logging_steps 1 \
    --logging_nan_inf_filter true