WANDB_PROJECT=specInfer python distill/train.py \
    --student_model_path eqhylxx/full-vicuna-160m \
    --teacher_model_path /rscratch/zhendong/lily/vicuna-7b-v1.3/  \
    --data_path data/code_search_net_train.json \
    --eval_data_path data/code_search_net_test.json \
    --bf16 True \
    --output_dir /rscratch/zhendong/lily/offline_code-search-python \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_steps 10 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --run_name offline_code-search-python \
    --mode offline
