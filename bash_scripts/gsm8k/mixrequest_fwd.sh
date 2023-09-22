WANDB_PROJECT=specInfer python distill/train.py \
    --student_model_path JackFram/llama-160m \
    --teacher_model_path /data/vicuna-7b-v1.3/ \
    --data_path data/gsm8k_train_with_answer.json \
    --max_propose_num 5 \
    --bf16 True \
    --output_dir /data/gsm8k_mixrequest_fwd \
    --num_train_epochs 2 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --run_name gsm8k_mixrequest_fwd \
    --mode offline \
    --sample_source mix_request \
    --kl_method forward

mkdir output
python distill/experiment/compare_model.py \
       --data /home/lily/spec_new/data/gsm8k_test.json \
       --student /data/gsm8k_mixrequest_fwd   > output/gsm8k_mixtoken_acc.out