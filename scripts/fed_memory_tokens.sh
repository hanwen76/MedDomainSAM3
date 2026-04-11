## Training

CUDA_VISIBLE_DEVICES=1 python /home/zhanghanwen/sam3-main/scripts/train_federated_free_memory_tokens_fedavg.py \
  --dataset-root /mnt/diskB/zhw/Prostate_82_256_JPG/ \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --output-dir /home/zhanghanwen/text-sam3/Prostate/BIDMC/federated_tokens_memory_fedavg \
  --holdout-site BIDMC \
  --text-prompt "prostate" \
  --lr 1e-2 \
  --epochs 25

CUDA_VISIBLE_DEVICES=1 python /home/zhanghanwen/sam3-main/scripts/train_federated_free_memory_tokens_fedavg.py \
  --dataset-root /mnt/diskB/zhw/Prostate_82_256_JPG/ \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --output-dir /home/zhanghanwen/text-sam3/Prostate/I2CVB/federated_tokens_memory_fedavg \
  --holdout-site I2CVB \
  --text-prompt "prostate" \
  --lr 1e-2 \
  --epochs 25

CUDA_VISIBLE_DEVICES=1 python /home/zhanghanwen/sam3-main/scripts/train_federated_free_memory_tokens_fedavg.py \
  --dataset-root /mnt/diskB/zhw/Prostate_82_256_JPG/ \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --output-dir /home/zhanghanwen/text-sam3/Prostate/HK/federated_tokens_memory_fedavg \
  --holdout-site HK \
  --text-prompt "prostate" \
  --lr 1e-2 \
  --epochs 25

CUDA_VISIBLE_DEVICES=1 python /home/zhanghanwen/sam3-main/scripts/train_federated_free_memory_tokens_fedavg.py \
  --dataset-root /mnt/diskB/zhw/Prostate_82_256_JPG/ \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --output-dir /home/zhanghanwen/text-sam3/Prostate/ISBI/federated_tokens_memory_fedavg \
  --holdout-site ISBI \
  --text-prompt "prostate" \
  --lr 1e-2 \
  --epochs 25

CUDA_VISIBLE_DEVICES=1 python /home/zhanghanwen/sam3-main/scripts/train_federated_free_memory_tokens_fedavg.py \
  --dataset-root /mnt/diskB/zhw/Prostate_82_256_JPG/ \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --output-dir /home/zhanghanwen/text-sam3/Prostate/ISBI_1.5/federated_tokens_memory_fedavg \
  --holdout-site ISBI_1.5 \
  --text-prompt "prostate" \
  --lr 1e-2 \
  --epochs 25

CUDA_VISIBLE_DEVICES=1 python /home/zhanghanwen/sam3-main/scripts/train_federated_free_memory_tokens_fedavg.py \
  --dataset-root /mnt/diskB/zhw/Prostate_82_256_JPG/ \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --output-dir /home/zhanghanwen/text-sam3/Prostate/UCL/federated_tokens_memory_fedavg \
  --holdout-site UCL \
  --text-prompt "prostate" \
  --lr 1e-2 \
  --epochs 25

## Evaluation

CUDA_VISIBLE_DEVICES=0 python /home/zhanghanwen/sam3-main/scripts/eval_federated_free_memory_tokens_fedavg.py \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --fedavg-model /home/zhanghanwen/text-sam3/Prostate/BIDMC/federated_tokens_memory_fedavg/BIDMC/fedavg_free_memory_tokens.pt \
  --dataset-root /mnt/diskB/zhw/Prostate_82_256_JPG/ \
  --holdout-site BIDMC \
  --output-dir /home/zhanghanwen/text-sam3/Prostate/BIDMC/federated_tokens_memory_fedavg/eval_out \
  --text-prompt "prostate"

CUDA_VISIBLE_DEVICES=0 python /home/zhanghanwen/sam3-main/scripts/eval_federated_free_memory_tokens_fedavg.py \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --fedavg-model /home/zhanghanwen/text-sam3/Prostate/I2CVB/federated_tokens_memory_fedavg/I2CVB/fedavg_free_memory_tokens.pt \
  --dataset-root /mnt/diskB/zhw/Prostate_82_256_JPG/ \
  --holdout-site I2CVB \
  --output-dir /home/zhanghanwen/text-sam3/Prostate/I2CVB/federated_tokens_memory_fedavg/eval_out \
  --text-prompt "prostate"

CUDA_VISIBLE_DEVICES=0 python /home/zhanghanwen/sam3-main/scripts/eval_federated_free_memory_tokens_fedavg.py \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --fedavg-model /home/zhanghanwen/text-sam3/Prostate/HK/federated_tokens_memory_fedavg/HK/fedavg_free_memory_tokens.pt \
  --dataset-root /mnt/diskB/zhw/Prostate_82_256_JPG/ \
  --holdout-site HK \
  --output-dir /home/zhanghanwen/text-sam3/Prostate/HK/federated_tokens_memory_fedavg/eval_out \
  --text-prompt "prostate"

CUDA_VISIBLE_DEVICES=0 python /home/zhanghanwen/sam3-main/scripts/eval_federated_free_memory_tokens_fedavg.py \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --fedavg-model /home/zhanghanwen/text-sam3/Prostate/ISBI/federated_tokens_memory_fedavg/ISBI/fedavg_free_memory_tokens.pt \
  --dataset-root /mnt/diskB/zhw/Prostate_82_256_JPG/ \
  --holdout-site ISBI \
  --output-dir /home/zhanghanwen/text-sam3/Prostate/ISBI/federated_tokens_memory_fedavg/eval_out \
  --text-prompt "prostate"

CUDA_VISIBLE_DEVICES=2 python /home/zhanghanwen/sam3-main/scripts/eval_federated_free_memory_tokens_fedavg.py \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --fedavg-model /home/zhanghanwen/text-sam3/Prostate/ISBI_1.5/federated_tokens_memory_fedavg/ISBI_1.5/fedavg_free_memory_tokens.pt \
  --dataset-root /mnt/diskB/zhw/Prostate_82_256_JPG/ \
  --holdout-site ISBI_1.5 \
  --output-dir /home/zhanghanwen/text-sam3/Prostate/ISBI_1.5/federated_tokens_memory_fedavg/eval_out \
  --text-prompt "prostate"

CUDA_VISIBLE_DEVICES=2 python /home/zhanghanwen/sam3-main/scripts/eval_federated_free_memory_tokens_fedavg.py \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --fedavg-model /home/zhanghanwen/text-sam3/Prostate/UCL/federated_tokens_memory_fedavg/UCL/fedavg_free_memory_tokens.pt \
  --dataset-root /mnt/diskB/zhw/Prostate_82_256_JPG/ \
  --holdout-site UCL \
  --output-dir /home/zhanghanwen/text-sam3/Prostate/UCL/federated_tokens_memory_fedavg/eval_out \
  --text-prompt "prostate"