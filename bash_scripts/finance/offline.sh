datapath=$1
sample=$2
kl=$3

python distill/train.py \
    --student_model_path JackFram/llama-160m \
    --teacher_model_path lmsys/vicuna-7b-v1.3/ \
    --data_path data/raw_data/gbharti_finance-alpace_train_with_answer.json \
    --eval_data_path data/raw_data/gbharti_finance-alpace_test.json \
    --max_propose_num 5 \
    --bf16 True \
    --output_dir $datapath/gbharti_finance-alpace_${sample}_${kl} \
    --num_train_epochs 2 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "epoch" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 256 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --run_name gbharti_finance-alpace_${sample}_${kl}\
    --mode offline \
    --sample_source $sample \
    --kl_method $kl