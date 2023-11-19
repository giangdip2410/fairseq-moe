PATH_TO_DATA="/home/gtruong/Project/ICML/Fair_MOE/fairseq-moe/preprocess_scripts/data-bin/wikitext-2"
jobname="stable_brainformer_lm_wt2_tiny_test"

CUDA_VISIBLE_DEVICES=0 fairseq-eval-lm ${PATH_TO_DATA} \
    --path /home/gtruong/Project/ICML/Fair_MOE/fairseq-moe/train_scripts/StableMOE/checkpoints/$jobname/checkpoint_best.pt \
    --batch-size 128 \
    --user-dir stablemoe \
    --ddp-backend=no_c10d \
    --seed 2410 \
    --tokens-per-sample 128 \
    --is-moe

