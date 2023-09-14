WANDB_PROJECT=specInfer python distill/train.py \
    --student_model_path eqhylxx/full-vicuna-160m \
    --teacher_model_path /data/vicuna-7b-v1.3/ \
    --data_path data/spider_train.json \
    --eval_data_path data/spider_eval.json \
    --max_propose_num 5 \
    --bf16 True \
    --output_dir /data/fix_student_kl_teacher_student_no-sample-grad_spider \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 10 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --run_name fix_student_kl_teacher_student_no-sample-grad_spider
