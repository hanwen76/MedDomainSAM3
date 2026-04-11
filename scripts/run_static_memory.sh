
CUDA_VISIBLE_DEVICES=0 python /home/zhanghanwen/sam3-main/scripts/build_static_memory_bank.py \
  --image-dir /mnt/diskB/zhw/Prostate_82_256_JPG/BIDMC/data_npy \
  --mask-dir /mnt/diskB/zhw/Prostate_82_256_JPG/BIDMC/label_npy \
  --output-path /home/zhanghanwen/text-sam3/Prostate/BIDMC/static_memory_bank.pt \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --default-text-prompt "prostate" \
  --prototype-count 16 \
  --prototype-grouping global

CUDA_VISIBLE_DEVICES=0 python /home/zhanghanwen/sam3-main/scripts/eval_medical_static_memory.py \
  --image-dir /mnt/diskB/zhw/Prostate_82_256_JPG/BIDMC/val_data_npy \
  --mask-dir /mnt/diskB/zhw/Prostate_82_256_JPG/BIDMC/val_label_npy \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --static-memory-bank-path /home/zhanghanwen/text-sam3/Prostate/BIDMC/static_memory_bank.pt \
  --output-dir /home/zhanghanwen/text-sam3/Prostate/BIDMC/eval_out \
  --text-prompt "prostate" \
  --static-memory-topk 1 \
  --no-object-threshold 0.2 \
  --memory-trigger-threshold 0.8 \
  --save-predictions

CUDA_VISIBLE_DEVICES=0 python /home/zhanghanwen/sam3-main/scripts/build_static_memory_bank.py \
  --image-dir /mnt/diskB/zhw/Prostate_82_256_JPG/HK/data_npy \
  --mask-dir /mnt/diskB/zhw/Prostate_82_256_JPG/HK/label_npy \
  --output-path /home/zhanghanwen/text-sam3/Prostate/HK/static_memory_bank.pt \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --default-text-prompt "prostate" \
  --prototype-count 16 \
  --prototype-grouping global

CUDA_VISIBLE_DEVICES=0 python /home/zhanghanwen/sam3-main/scripts/eval_medical_static_memory.py \
  --image-dir /mnt/diskB/zhw/Prostate_82_256_JPG/HK/val_data_npy \
  --mask-dir /mnt/diskB/zhw/Prostate_82_256_JPG/HK/val_label_npy \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --static-memory-bank-path /home/zhanghanwen/text-sam3/Prostate/HK/static_memory_bank.pt \
  --output-dir /home/zhanghanwen/text-sam3/Prostate/HK/eval_out \
  --text-prompt "prostate" \
  --static-memory-topk 1 \
  --no-object-threshold 0.2 \
  --memory-trigger-threshold 0.8 \
  --save-predictions

CUDA_VISIBLE_DEVICES=0 python /home/zhanghanwen/sam3-main/scripts/build_static_memory_bank.py \
  --image-dir /mnt/diskB/zhw/Prostate_82_256_JPG/I2CVB/data_npy \
  --mask-dir /mnt/diskB/zhw/Prostate_82_256_JPG/I2CVB/label_npy \
  --output-path /home/zhanghanwen/text-sam3/Prostate/I2CVB/static_memory_bank.pt \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --default-text-prompt "prostate" \
  --prototype-count 16 \
  --prototype-grouping global

CUDA_VISIBLE_DEVICES=0 python /home/zhanghanwen/sam3-main/scripts/eval_medical_static_memory.py \
  --image-dir /mnt/diskB/zhw/Prostate_82_256_JPG/I2CVB/val_data_npy \
  --mask-dir /mnt/diskB/zhw/Prostate_82_256_JPG/I2CVB/val_label_npy \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --static-memory-bank-path /home/zhanghanwen/text-sam3/Prostate/I2CVB/static_memory_bank.pt \
  --output-dir /home/zhanghanwen/text-sam3/Prostate/I2CVB/eval_out \
  --text-prompt "prostate" \
  --static-memory-topk 1 \
  --no-object-threshold 0.2 \
  --memory-trigger-threshold 0.8 \
  --save-predictions

CUDA_VISIBLE_DEVICES=0 python /home/zhanghanwen/sam3-main/scripts/build_static_memory_bank.py \
  --image-dir /mnt/diskB/zhw/Prostate_82_256_JPG/ISBI/data_npy \
  --mask-dir /mnt/diskB/zhw/Prostate_82_256_JPG/ISBI/label_npy \
  --output-path /home/zhanghanwen/text-sam3/Prostate/ISBI/static_memory_bank.pt \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --default-text-prompt "prostate" \
  --prototype-count 16 \
  --prototype-grouping global

CUDA_VISIBLE_DEVICES=0 python /home/zhanghanwen/sam3-main/scripts/eval_medical_static_memory.py \
  --image-dir /mnt/diskB/zhw/Prostate_82_256_JPG/ISBI/val_data_npy \
  --mask-dir /mnt/diskB/zhw/Prostate_82_256_JPG/ISBI/val_label_npy \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --static-memory-bank-path /home/zhanghanwen/text-sam3/Prostate/ISBI/static_memory_bank.pt \
  --output-dir /home/zhanghanwen/text-sam3/Prostate/ISBI/eval_out \
  --text-prompt "prostate" \
  --static-memory-topk 1 \
  --no-object-threshold 0.2 \
  --memory-trigger-threshold 0.8 \
  --save-predictions

CUDA_VISIBLE_DEVICES=0 python /home/zhanghanwen/sam3-main/scripts/build_static_memory_bank.py \
  --image-dir /mnt/diskB/zhw/Prostate_82_256_JPG/ISBI_1.5/data_npy \
  --mask-dir /mnt/diskB/zhw/Prostate_82_256_JPG/ISBI_1.5/label_npy \
  --output-path /home/zhanghanwen/text-sam3/Prostate/ISBI_1.5/static_memory_bank.pt \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --default-text-prompt "prostate" \
  --prototype-count 16 \
  --prototype-grouping global

CUDA_VISIBLE_DEVICES=0 python /home/zhanghanwen/sam3-main/scripts/eval_medical_static_memory.py \
  --image-dir /mnt/diskB/zhw/Prostate_82_256_JPG/ISBI_1.5/val_data_npy \
  --mask-dir /mnt/diskB/zhw/Prostate_82_256_JPG/ISBI_1.5/val_label_npy \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --static-memory-bank-path /home/zhanghanwen/text-sam3/Prostate/ISBI_1.5/static_memory_bank.pt \
  --output-dir /home/zhanghanwen/text-sam3/Prostate/ISBI_1.5/eval_out \
  --text-prompt "prostate" \
  --static-memory-topk 1 \
  --no-object-threshold 0.2 \
  --memory-trigger-threshold 0.8 \
  --save-predictions

CUDA_VISIBLE_DEVICES=0 python /home/zhanghanwen/sam3-main/scripts/build_static_memory_bank.py \
  --image-dir /mnt/diskB/zhw/Prostate_82_256_JPG/UCL/data_npy \
  --mask-dir /mnt/diskB/zhw/Prostate_82_256_JPG/UCL/label_npy \
  --output-path /home/zhanghanwen/text-sam3/Prostate/UCL/static_memory_bank.pt \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --default-text-prompt "prostate" \
  --prototype-count 16 \
  --prototype-grouping global

CUDA_VISIBLE_DEVICES=0 python /home/zhanghanwen/sam3-main/scripts/eval_medical_static_memory.py \
  --image-dir /mnt/diskB/zhw/Prostate_82_256_JPG/UCL/val_data_npy \
  --mask-dir /mnt/diskB/zhw/Prostate_82_256_JPG/UCL/val_label_npy \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --static-memory-bank-path /home/zhanghanwen/text-sam3/Prostate/UCL/static_memory_bank.pt \
  --output-dir /home/zhanghanwen/text-sam3/Prostate/UCL/eval_out \
  --text-prompt "prostate" \
  --static-memory-topk 1 \
  --no-object-threshold 0.2 \
  --memory-trigger-threshold 0.8 \
  --save-predictions

'''
# brain tumor
CUDA_VISIBLE_DEVICES=0 python /home/zhanghanwen/sam3-main/scripts/build_static_memory_bank.py \
  --image-dir /mnt/diskB/zhw/FeTS2022_FedDG_82_240_2_JPG/1/data_npy \
  --mask-dir /mnt/diskB/zhw/FeTS2022_FedDG_82_240_2_JPG/1/label_npy \
  --output-path /home/zhanghanwen/text-sam3/FeTS/1/static_memory_bank.pt \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --default-text-prompt "brain tumor" \
  --prototype-count 8 \
  --prototype-grouping global

CUDA_VISIBLE_DEVICES=0 python /home/zhanghanwen/sam3-main/scripts/eval_medical_static_memory.py \
  --image-dir /mnt/diskB/zhw/FeTS2022_FedDG_82_240_2_JPG/1/val_data_npy \
  --mask-dir /mnt/diskB/zhw/FeTS2022_FedDG_82_240_2_JPG/1/val_label_npy \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --static-memory-bank-path /home/zhanghanwen/text-sam3/FeTS/1/static_memory_bank.pt \
  --output-dir /home/zhanghanwen/text-sam3/FeTS/1/eval_out \
  --text-prompt "brain tumor" \
  --static-memory-topk 2 \
  --no-object-threshold 0.2 \
  --memory-trigger-threshold 0.8 \
  --save-predictions
'''
'''
# federated static memory
CUDA_VISIBLE_DEVICES=0 python /home/zhanghanwen/sam3-main/scripts/run_federated_static_memory.py \
  --dataset-root /mnt/diskB/zhw/Prostate_82_256_JPG \
  --holdout-site BIDMC \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --output-dir /home/zhanghanwen/text-sam3/Prostate/BIDMC/fed_static_memory_out \
  --text-prompt "prostate" \
  --device cuda \
  --prototype-count 8 \
  --local-prototype-count 4 \
  --static-memory-topk 1 \
  --no-object-threshold 0.2 \
  --memory-trigger-threshold 0.8

CUDA_VISIBLE_DEVICES=0 python /home/zhanghanwen/sam3-main/scripts/run_federated_static_memory.py \
  --dataset-root /mnt/diskB/zhw/Prostate_82_256_JPG \
  --holdout-site I2CVB \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --output-dir /home/zhanghanwen/text-sam3/Prostate/I2CVB/fed_static_memory_out \
  --text-prompt "prostate" \
  --device cuda \
  --prototype-count 8 \
  --local-prototype-count 4 \
  --static-memory-topk 1 \
  --no-object-threshold 0.2 \
  --memory-trigger-threshold 0.8

CUDA_VISIBLE_DEVICES=1 python /home/zhanghanwen/sam3-main/scripts/run_federated_static_memory.py \
  --dataset-root /mnt/diskB/zhw/Prostate_82_256_JPG \
  --holdout-site HK \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --output-dir /home/zhanghanwen/text-sam3/Prostate/HK/fed_static_memory_out \
  --text-prompt "prostate" \
  --device cuda \
  --prototype-count 8 \
  --local-prototype-count 4 \
  --static-memory-topk 1 \
  --no-object-threshold 0.2 \
  --memory-trigger-threshold 0.8
  '''