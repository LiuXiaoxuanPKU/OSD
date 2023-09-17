trial=$1              # trial number: 1, 2, 3, 4, 5, 6, ...

export WANDB_RUN_GROUP=lily-falcon
export WANDB_PROJECT=specInfer
export CUDA_LAUNCH_BLOCKING=1
#export CUDA_VISIBLE_DEVICES=0

python3 -m torch.distributed.run --nproc_per_node=8 \
    --master_port 20725 \
    distill/train_mlm.py \
    --student_model_path google/t5-efficient-small  \
    --teacher_model_path google/t5-efficient-xl \
    --dataset_name wmt16 \
    --dataset_config de-en \
    --is_qa True \
    --do_train \
    --do_eval \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_steps 5 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.02 \
    --logging_steps 1 \
    --source_max_length 1024 \
    --train_target_max_length 512 \
    --val_target_max_length 512 \
    --test_target_max_length 512 \
    --output_dir /home/lanxiang/MIT/LLMs_and_TVM/specd/specNBCE-main/flan_wmt_seqlen512 2>&1 | tee logs/train/flan_wmt14_flan_xl_to_small_trial_${trial}.log
