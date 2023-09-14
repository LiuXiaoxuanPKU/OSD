trial=$1              # trial number: 1, 2, 3, 4, 5, 6, ...

export WANDB_PROJECT=specInfer_t5
export CUDA_LAUNCH_BLOCKING=1

python3 -m torch.distributed.run --nproc_per_node=8 \
    --master_port 20725 \
    distill/train_mlm.py \
    --student_model_path google/t5-efficient-small  \
    --teacher_model_path google/t5-efficient-xl \
    --dataset_name gsm8k \
    --do_train \
    --do_eval \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --source_max_length 1024 \
    --train_target_max_length 64 \
    --val_target_max_length 128 \
    --test_target_max_length 128 \
    --output_dir /home/lanxiang/MIT/LLMs_and_TVM/specd/specNBCE-main/mlm_xsum 2>&1 | tee logs/train/xsum_t5_xl_to_small_trial_${trial}.log
