#!/bin/sh
set -x
set -e

export PYTHONPATH="./:$PYTHONPATH"

path_data='/path/to/data/rovi/data/JPEGImages'
path_inpaint_data='/path/to/data/rovi/data/InpaintImages/'
path_mask_data='/path/to/data/rovi/data/Annotations'
video_token_len=622



basic_arguments="--version v1 \
        --bf16 True \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 1 \
        --dataloader_num_workers 0 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy 'no' \
        --save_strategy 'steps' \
        --save_steps 10000 \
        --save_total_limit 3 \
        --num_train_epochs 50 \
        --learning_rate 2e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type 'cosine' \
        --logging_steps 1 \
        --model_max_length 1024 \
        --gradient_checkpointing True \
        --lazy_preprocess True \
        --mm_use_vid_start_end True \
        --tune_mm_mlp_adapter True \
        --freeze_backbone True \
        --lora_enable"

path_llava='/path/to/weights/llava-v1.5-7b/'
path_projection='/path/to/weights/projection/mm_projector_7b_1.5_336px.bin'
path_data='/path/to/data/video_editing_arxiv/for_caption.json' # <path to filtered_336_video_chatgpt_training.json>

path_video_feats='/path/to/weights/video_feats'
path_inpainted_video_feats='/path/to/weights/video_inpainted_feats'
inpainted_prompt_path='/path/to/data/gt_layout_prediction.json'
path_output='_ckpt/raccoon_v2p_train_output'

# Multi-object description
# Single-object description
# Layout Prediction
CUDA_LAUNCH_BLOCKING=1 \
WANDB_DISABLED='true' torchrun --nnodes=1 --nproc_per_node=$1 --master_port=7235 \
        LLaVA/llava/train/train_raccoon.py \
        --model_name_or_path $path_llava \
        --data_path $path_data \
        --inpainted_data_path $path_inpaint_data\
        --inpainted_prompt_path $inpainted_prompt_path \
        --output_dir $path_output \
        --pretrain_mm_mlp_adapter $path_projection \
        --video_folder $path_video_feats \
        --inpainted_video_folder $path_inpainted_video_feats \
        --video_token_len $video_token_len \        
        $basic_arguments
        
# # Single-object description
# CUDA_LAUNCH_BLOCKING=1 \
# WANDB_DISABLED='true' torchrun --nnodes=1 --nproc_per_node=$1 --master_port=7235 \
#         LLaVA/llava/train/train_raccoon.py \
#         --model_name_or_path $path_llava \
#         --data_path $path_data \
#         --inpainted_data_path $path_inpaint_data\
#         --inpainted_prompt_path $inpainted_prompt_path \
#         --output_dir $path_output \
#         --pretrain_mm_mlp_adapter $path_projection \
#         --video_folder $path_video_feats \
#         --inpainted_video_folder $path_inpainted_video_feats \
#         --video_token_len $video_token_len \        
#         $basic_arguments
        
# # Layout Prediction
# CUDA_LAUNCH_BLOCKING=1 \
# WANDB_DISABLED='true' torchrun --nnodes=1 --nproc_per_node=$1 --master_port=7235 \
#         LLaVA/llava/train/train_raccoon.py \
#         --model_name_or_path $path_llava \
#         --data_path $path_data \
#         --inpainted_data_path $path_inpaint_data\
#         --inpainted_prompt_path $inpainted_prompt_path \
#         --output_dir $path_output \
#         --pretrain_mm_mlp_adapter $path_projection \
#         --video_folder $path_video_feats \
#         --inpainted_video_folder $path_inpainted_video_feats \
#         --video_token_len $video_token_len \        
#         $basic_arguments
        