export CUDA_VISIBLE_DEVICES=0

datapath=$1
trial=$2

torchrun --nproc_per_node=1 predictor/train_predictor.py \
    --model_name_or_path /home/hedgehog/workspace/models/vicuna-160m \
    --teacher_model_name_or_path /home/hedgehog/workspace/models/vicuna-7b-v1.5 \
    --load_in_4bit True \
    --data_path /home/hedgehog/workspace/OSD/data/vicuna160m_chatbot_arena_all_token_acceptance_rate_for_training_temp_0p001_1k1.json \
    --bf16 True \
    --output_dir ${datapath}/vicuna160m_layer1_predictor_trial_${trial} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 6 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 2 \
    --learning_rate 1e-3 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --lazy_preprocess True \
    --predictor_num_heads 1 \
    --predictor_num_layers 1