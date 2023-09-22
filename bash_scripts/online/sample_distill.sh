datapath=$1

WANDB_PROJECT=specInfer python distill/train.py \
    --student_model_path JackFram/llama-160m \
    --teacher_model_path $datapath/vicuna-7b-v1.3/ \
    --data_path data/online_sample.json \
    --max_propose_num 5 \
    --bf16 True \
    --output_dir $datapath/online_sample \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "consine" \
    --tf32 True \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --run_name online_sample_distill \
    --mode offline \
    --sample_source teacher \
    --kl_method forward