# locally train free memory tokens for each site and evaluate on each site

CUDA_VISIBLE_DEVICES=2 python /home/zhanghanwen/sam3-main/scripts/train_free_memory_tokens.py \
  --image-dir /mnt/diskB/zhw/Prostate_82_256_JPG/BIDMC/data_npy \
  --mask-dir /mnt/diskB/zhw/Prostate_82_256_JPG/BIDMC/label_npy \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --output-path /home/zhanghanwen/text-sam3/Prostate/BIDMC/free_memory_tokens.pt \
  --text-prompt "prostate" \
  --device cuda \
  --num-tokens 4 \
  --epochs 25 \
  --lr 1e-2

CUDA_VISIBLE_DEVICES=2 python /home/zhanghanwen/sam3-main/scripts/eval_medical_static_memory.py \
  --image-dir /mnt/diskB/zhw/Prostate_82_256_JPG/BIDMC/val_data_npy \
  --mask-dir /mnt/diskB/zhw/Prostate_82_256_JPG/BIDMC/val_label_npy \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --free-memory-ckpt /home/zhanghanwen/text-sam3/Prostate/BIDMC/free_memory_tokens.pt \
  --output-dir /home/zhanghanwen/text-sam3/Prostate/BIDMC/eval_free_memory \
  --device cuda \
  --prompt-mode text \
  --text-prompt "prostate" \
  --free-memory-num-tokens 4 \
  --no-object-threshold 0.2 \
  --memory-trigger-threshold 0.8 \
  --save-predictions

CUDA_VISIBLE_DEVICES=2 python /home/zhanghanwen/sam3-main/scripts/train_free_memory_tokens.py \
  --image-dir /mnt/diskB/zhw/Prostate_82_256_JPG/I2CVB/data_npy \
  --mask-dir /mnt/diskB/zhw/Prostate_82_256_JPG/I2CVB/label_npy \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --output-path /home/zhanghanwen/text-sam3/Prostate/I2CVB/free_memory_tokens.pt \
  --text-prompt "prostate" \
  --device cuda \
  --num-tokens 4 \
  --epochs 25 \
  --lr 1e-2

CUDA_VISIBLE_DEVICES=2 python /home/zhanghanwen/sam3-main/scripts/eval_medical_static_memory.py \
  --image-dir /mnt/diskB/zhw/Prostate_82_256_JPG/I2CVB/val_data_npy \
  --mask-dir /mnt/diskB/zhw/Prostate_82_256_JPG/I2CVB/val_label_npy \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --free-memory-ckpt /home/zhanghanwen/text-sam3/Prostate/I2CVB/free_memory_tokens.pt \
  --output-dir /home/zhanghanwen/text-sam3/Prostate/I2CVB/eval_free_memory \
  --device cuda \
  --prompt-mode text \
  --text-prompt "prostate" \
  --free-memory-num-tokens 4 \
  --no-object-threshold 0.2 \
  --memory-trigger-threshold 0.8 \
  --save-predictions

CUDA_VISIBLE_DEVICES=2 python /home/zhanghanwen/sam3-main/scripts/train_free_memory_tokens.py \
  --image-dir /mnt/diskB/zhw/Prostate_82_256_JPG/HK/data_npy \
  --mask-dir /mnt/diskB/zhw/Prostate_82_256_JPG/HK/label_npy \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --output-path /home/zhanghanwen/text-sam3/Prostate/HK/free_memory_tokens.pt \
  --text-prompt "prostate" \
  --device cuda \
  --num-tokens 4 \
  --epochs 25 \
  --lr 1e-2

CUDA_VISIBLE_DEVICES=2 python /home/zhanghanwen/sam3-main/scripts/eval_medical_static_memory.py \
  --image-dir /mnt/diskB/zhw/Prostate_82_256_JPG/HK/val_data_npy \
  --mask-dir /mnt/diskB/zhw/Prostate_82_256_JPG/HK/val_label_npy \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --free-memory-ckpt /home/zhanghanwen/text-sam3/Prostate/HK/free_memory_tokens.pt \
  --output-dir /home/zhanghanwen/text-sam3/Prostate/HK/eval_free_memory \
  --device cuda \
  --prompt-mode text \
  --text-prompt "prostate" \
  --free-memory-num-tokens 4 \
  --no-object-threshold 0.2 \
  --memory-trigger-threshold 0.8 \
  --save-predictions

CUDA_VISIBLE_DEVICES=2 python /home/zhanghanwen/sam3-main/scripts/train_free_memory_tokens.py \
  --image-dir /mnt/diskB/zhw/Prostate_82_256_JPG/ISBI/data_npy \
  --mask-dir /mnt/diskB/zhw/Prostate_82_256_JPG/ISBI/label_npy \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --output-path /home/zhanghanwen/text-sam3/Prostate/ISBI/free_memory_tokens.pt \
  --text-prompt "prostate" \
  --device cuda \
  --num-tokens 4 \
  --epochs 25 \
  --lr 1e-2

CUDA_VISIBLE_DEVICES=2 python /home/zhanghanwen/sam3-main/scripts/eval_medical_static_memory.py \
  --image-dir /mnt/diskB/zhw/Prostate_82_256_JPG/ISBI/val_data_npy \
  --mask-dir /mnt/diskB/zhw/Prostate_82_256_JPG/ISBI/val_label_npy \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --free-memory-ckpt /home/zhanghanwen/text-sam3/Prostate/ISBI/free_memory_tokens.pt \
  --output-dir /home/zhanghanwen/text-sam3/Prostate/ISBI/eval_free_memory \
  --device cuda \
  --prompt-mode text \
  --text-prompt "prostate" \
  --free-memory-num-tokens 4 \
  --no-object-threshold 0.2 \
  --memory-trigger-threshold 0.8 \
  --save-predictions

CUDA_VISIBLE_DEVICES=2 python /home/zhanghanwen/sam3-main/scripts/train_free_memory_tokens.py \
  --image-dir /mnt/diskB/zhw/Prostate_82_256_JPG/ISBI_1.5/data_npy \
  --mask-dir /mnt/diskB/zhw/Prostate_82_256_JPG/ISBI_1.5/label_npy \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --output-path /home/zhanghanwen/text-sam3/Prostate/ISBI_1.5/free_memory_tokens.pt \
  --text-prompt "prostate" \
  --device cuda \
  --num-tokens 4 \
  --epochs 25 \
  --lr 1e-2

CUDA_VISIBLE_DEVICES=2 python /home/zhanghanwen/sam3-main/scripts/eval_medical_static_memory.py \
  --image-dir /mnt/diskB/zhw/Prostate_82_256_JPG/ISBI_1.5/val_data_npy \
  --mask-dir /mnt/diskB/zhw/Prostate_82_256_JPG/ISBI_1.5/val_label_npy \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --free-memory-ckpt /home/zhanghanwen/text-sam3/Prostate/ISBI_1.5/free_memory_tokens.pt \
  --output-dir /home/zhanghanwen/text-sam3/Prostate/ISBI_1.5/eval_free_memory \
  --device cuda \
  --prompt-mode text \
  --text-prompt "prostate" \
  --free-memory-num-tokens 4 \
  --no-object-threshold 0.2 \
  --memory-trigger-threshold 0.8 \
  --save-predictions

CUDA_VISIBLE_DEVICES=2 python /home/zhanghanwen/sam3-main/scripts/train_free_memory_tokens.py \
  --image-dir /mnt/diskB/zhw/Prostate_82_256_JPG/UCL/data_npy \
  --mask-dir /mnt/diskB/zhw/Prostate_82_256_JPG/UCL/label_npy \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --output-path /home/zhanghanwen/text-sam3/Prostate/UCL/free_memory_tokens.pt \
  --text-prompt "prostate" \
  --device cuda \
  --num-tokens 4 \
  --epochs 25 \
  --lr 1e-2

CUDA_VISIBLE_DEVICES=2 python /home/zhanghanwen/sam3-main/scripts/eval_medical_static_memory.py \
  --image-dir /mnt/diskB/zhw/Prostate_82_256_JPG/UCL/val_data_npy \
  --mask-dir /mnt/diskB/zhw/Prostate_82_256_JPG/UCL/val_label_npy \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --free-memory-ckpt /home/zhanghanwen/text-sam3/Prostate/UCL/free_memory_tokens.pt \
  --output-dir /home/zhanghanwen/text-sam3/Prostate/UCL/eval_free_memory \
  --device cuda \
  --prompt-mode text \
  --text-prompt "prostate" \
  --free-memory-num-tokens 4 \
  --no-object-threshold 0.2 \
  --memory-trigger-threshold 0.8 \
  --save-predictions

'''

# Brain Tumor

CUDA_VISIBLE_DEVICES=1 python /home/zhanghanwen/sam3-main/scripts/train_free_memory_tokens.py \
  --image-dir /mnt/diskB/zhw/FeTS2022_FedDG_82_240_2_JPG/1/data_npy \
  --mask-dir /mnt/diskB/zhw/FeTS2022_FedDG_82_240_2_JPG/1/label_npy \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --output-path /home/zhanghanwen/text-sam3/FeTS/1/free_memory_tokens.pt \
  --text-prompt "brain tumor" \
  --device cuda \
  --num-tokens 4 \
  --epochs 25 \
  --lr 1e-2

CUDA_VISIBLE_DEVICES=1 python /home/zhanghanwen/sam3-main/scripts/eval_medical_static_memory.py \
  --image-dir /mnt/diskB/zhw/FeTS2022_FedDG_82_240_2_JPG/1/val_data_npy \
  --mask-dir /mnt/diskB/zhw/FeTS2022_FedDG_82_240_2_JPG/1/val_label_npy \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --free-memory-ckpt /home/zhanghanwen/text-sam3/FeTS/1/free_memory_tokens.pt \
  --output-dir /home/zhanghanwen/text-sam3/FeTS/1/eval_free_memory \
  --device cuda \
  --prompt-mode text \
  --text-prompt "brain tumor" \
  --free-memory-num-tokens 4 \
  --no-object-threshold 0.2 \
  --memory-trigger-threshold 0.8 \
  --save-predictions
'''

# Cervical Cancer
# Somthing went wrong with the dataloader

CUDA_VISIBLE_DEVICES=3 python /home/zhanghanwen/sam3-main/scripts/train_free_memory_tokens.py \
  --image-dir /home/zhanghanwen/fundus_1024_256_cup/fundus1/data_npy \
  --mask-dir /home/zhanghanwen/fundus_1024_256_cup/fundus1/label_npy \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --output-path /home/zhanghanwen/text-sam3/fundus_1024_256_cup/fundus1/free_memory_tokens.pt \
  --text-prompt "fundus cup" \
  --device cuda \
  --num-tokens 4 \
  --epochs 25 \
  --lr 1e-2

CUDA_VISIBLE_DEVICES=3 python /home/zhanghanwen/sam3-main/scripts/eval_medical_static_memory.py \
  --image-dir /home/zhanghanwen/fundus_1024_256_cup/fundus1/val_data_npy \
  --mask-dir /home/zhanghanwen/fundus_1024_256_cup/fundus1/val_label_npy \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --free-memory-ckpt /home/zhanghanwen/text-sam3/fundus_1024_256_cup/fundus1/free_memory_tokens.pt \
  --output-dir /home/zhanghanwen/text-sam3/fundus_1024_256_cup/fundus1/eval_free_memory \
  --device cuda \
  --prompt-mode text \
  --text-prompt "fundus cup" \
  --free-memory-num-tokens 4 \
  --no-object-threshold 0.2 \
  --memory-trigger-threshold 0.8 \
  --save-predictions