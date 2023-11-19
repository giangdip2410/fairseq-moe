PATH_TO_DATA="/home/gtruong/Project/ICML/Fair_MOE/fairseq-moe/preprocess_scripts/data-bin/wikitext-2"
jobname="xmoe_brainformer_lm_wt2_small"

CUDA_VISIBLE_DEVICES=0 fairseq-eval-lm ${PATH_TO_DATA} \
    --path /home/gtruong/Project/ICML/Fair_MOE/fairseq-moe/train_scripts/XMOE/checkpoints/$jobname/checkpoint_best.pt \
    --gen-subset valid \
    --sample-break-mode none \
    --batch-size 1 \
    --user-dir xmoe \
    --ddp-backend=no_c10d \
    --seed 2410 \
    --distributed-world-size 1 \
    --is-moe

