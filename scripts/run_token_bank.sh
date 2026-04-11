CUDA_VISIBLE_DEVICES=1 python "/home/zhanghanwen/sam3-main/scripts/eval_memory_hybrid_ablation.py" \
  --image-dir /mnt/diskB/zhw/Prostate_82_256_JPG/BIDMC/data_npy \
  --mask-dir /mnt/diskB/zhw/Prostate_82_256_JPG/BIDMC/label_npy \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --static-memory-bank-path /home/zhanghanwen/text-sam3/Prostate/BIDMC/static_memory_bank.pt \
  --free-memory-ckpt /home/zhanghanwen/text-sam3/Prostate/BIDMC/free_memory_tokens.pt \
  --output-dir /home/zhanghanwen/text-sam3/Prostate/BIDMC/eval_hybrid_ablation \
  --text-prompt "prostate" \
  --device cuda \
  --use-gate \
  --no-object-threshold 0.2 \
  --memory-trigger-threshold 0.8 \
  --hybrid-fusion score_weighted

CUDA_VISIBLE_DEVICES=1 python3 /home/zhanghanwen/sam3-main/scripts/run_site_memory_auto.py \
  --dataset-root /mnt/diskB/zhw/Prostate_82_256_JPG \
  --site BIDMC \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --output-root /home/zhanghanwen/text-sam3/Prostate/BIDMC/hybrid_memory_auto \
  --text-prompt "prostate"

# --loader-mode sequential 会按 datasets-json 里给出的顺序，依次加载各部位数据进行训练
# 每个样本按自己的 prompt_id 只使用对应 token 组，所以是“部位独立优化 token”

CUDA_VISIBLE_DEVICES=1 python "/Users/zhanghanwen/Documents/New project/sam3-main/scripts/train_prompt_token_bank.py" \
  --datasets-json /path/to/datasets.json \
  --output-path /path/to/prompt_token_bank.pt \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --device cuda \
  --tokens-per-prompt 4 \
  --loader-mode sequential \
  --epochs 5 \
  --lr 1e-2

CUDA_VISIBLE_DEVICES=1 python "/Users/zhanghanwen/Documents/New project/sam3-main/scripts/eval_prompt_token_bank.py" \
  --image-dir /path/to/val_images \
  --mask-dir /path/to/val_masks \
  --checkpoint-path /home/zhanghanwen/checkpoints/sam3.pt \
  --token-bank-ckpt /path/to/prompt_token_bank.pt \
  --output-dir /path/to/eval_prompt_token_bank \
  --text-prompt "prostate lesion" \
  --device cuda \
  --no-object-threshold 0.2 \
  --memory-trigger-threshold 0.8