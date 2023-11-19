PATH_TO_DATA="/home/gtruong/Project/ICML/Fair_MOE/fairseq-moe/preprocess_scripts/data-bin/wikitext-2"
jobname="competemoe_brainformer_lm_wt2_small"

CUDA_VISIBLE_DEVICES=0 fairseq-eval-lm ${PATH_TO_DATA} \
    --path ../checkpoints/$jobname/checkpoint_best.pt \
    --batch-size 128 \
    --ddp-backend=no_c10d \
    --seed 2410 \
    --is-moe

