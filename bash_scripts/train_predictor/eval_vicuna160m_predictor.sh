export CUDA_VISIBLE_DEVICES=0

datapath=$1
trial=$2

torchrun --nproc_per_node=1 predictor/train_predictor.py \
    --model_name_or_path /home/hedgehog/workspace/models/vicuna-160m \
    --teacher_model_name_or_path /home/hedgehog/workspace/models/vicuna-7b-v1.5 \
    --load_in_4bit True \
    --do_train False \
    --do_eval True \
    --data_path /home/hedgehog/workspace/OSD/data/vicuna160m_chatbot_arena_all_token_acceptance_rate_for_training_temp_0p001_1k1.json \
    --eval_data_path /home/hedgehog/workspace/OSD/data/vicuna160m_chatbot_arena_all_token_acceptance_rate_prepare_for_training_eval_500_1.json \
    --predictor_head_name_or_path /home/hedgehog/workspace/OSD/outputs/vicuna160m_layer1_predictor_trial_1_predictor_mlp_vicuna-160m_predictor_1_lr_0.001_layers_1 \
    --bf16 True \
    --output_dir ${datapath}/eval_vicuna160m_layer1_predictor_trial_${trial} \
    --per_device_eval_batch_size 6 \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --lazy_preprocess True \
    --predictor_num_heads 1 \
    --predictor_num_layers 1