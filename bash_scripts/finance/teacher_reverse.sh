datapath=$1

WANDB_PROJECT=specInfer python distill/train.py \
    --student_model_path  $datapath/llama-160m \
    --teacher_model_path $datapath/vicuna-7b-v1.3/ \
    --data_path data/gbharti_finance-alpaca_train_with_answer.json \
    --eval_data_path data/gbharti_finance-alpaca_eval.json \
    --bf16 True \
    --output_dir $datapath/finance_teacher_reverse \
    --num_train_epochs 2 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "epoch" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 256 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --run_name finance_teacher_reverse \
    --mode offline \
    --sample_source teacher \
    --kl_method reverse

# mkdir output
# python distill/experiment/compare_model.py \
#        --data data/gbharti_finance-alpaca_eval.json \
#        --student $datapath/finance_teacher_reverse   > output/finance_teacher_reverse_acc.out