PATH_TO_DATA="/home/gtruong/Project/ICML/Fair_MOE/fairseq-moe/preprocess_scripts/data-bin/wikitext-2"
jobname="xmoe_brainformer_lm_wt2_large"
fairseq-eval-lm ${PATH_TO_DATA} \
    --path ../checkpoints/$jobname/checkpoint_best.pt \
    --batch-size 128 \
    --ddp-backend=no_c10d \
    --seed 2410 \
    --is-moe
