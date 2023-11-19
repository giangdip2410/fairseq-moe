PATH_TO_DATA="/home/gtruong/Project/ICML/Fair_MOE/fairseq-moe/preprocess_scripts/data/enwik8/data-bin"
jobname="xmoe_brainformer_lm_enwik8_large"
fairseq-eval-lm ${PATH_TO_DATA} \
    --path ../checkpoints/$jobname/checkpoint_best.pt \
    --batch-size 128 \
    --ddp-backend=no_c10d \
    --seed 2410 \
    --is-moe
