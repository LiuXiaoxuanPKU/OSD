trial=$1              # trial number: 1, 2, 3, 4, 5, 6, ...

export WANDB_RUN_GROUP=lily-falcon
export WANDB_PROJECT=specInfer
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=5

python3 -m torch.distributed.run --nproc_per_node=1 \
    --master_port 20739 \
    distill/train_mlm.py \
    --student_model_path google/t5-efficient-small  \
    --teacher_model_path google/flan-t5-xl \
    --dataset_name spider_with_answers \
    --max_propose_num 5 \
    --mode online \
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
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --source_max_length 1024 \
    --train_target_max_length 512 \
    --val_target_max_length 512 \
    --test_target_max_length 512 \
    --run_name onlinet5_spider_flanteacher_2e-5lr \
    --output_dir /home/lanxiang/MIT/LLMs_and_TVM/specd/specNBCE-main/outputs/t5_spider_online_flan_teacher_2e-5lr 2>&1 | tee logs/train/online_spider_t5_xl_to_small_trial_${trial}_flan_teacher_2e-5lr.log
