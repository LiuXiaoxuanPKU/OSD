datapath=$1
sample=$2
kl=$3

torchrun --nproc_per_node=2 distill/train.py \
    --student_model_path JackFram/llama-160m \
    --teacher_model_path lmsys/vicuna-7b-v1.3 \
    --data_path data/raw_data/spider_train_with_answer.json \
    --max_propose_num 5 \
    --bf16 True \
    --output_dir $datapath/spider_${sample}_${kl} \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 256 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --run_name spider_${sample}_${kl} \
    --mode offline \
    --sample_source $sample \
    --kl_method $kl
