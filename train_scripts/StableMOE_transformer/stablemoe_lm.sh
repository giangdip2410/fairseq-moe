PATH_TO_DATA="/home/gtruong/Project/ICML/Fair_MOE/fairseq-moe/preprocess_scripts/data/enwik8/data-bin"
jobname="stalemoe_transformer_lm"
fairseq-train --task language_modeling \
    ${PATH_TO_DATA} \
    --num-workers 16 \
    --activation-fn gelu \
    --share-decoder-input-output-embed \
    --validate-interval-updates 1000 \
    --save-interval-updates 1000 \
    --no-epoch-checkpoints \
    --memory-efficient-fp16 \
    --fp16-init-scale 4 \
    --save-dir ../checkpoints/$jobname \
    --user-dir stablemoe \
    --arch transformer_lm_stable_BaseGPT_x1_small \
    --moe-type base_layer \
    --two-stage-updates 6000 \
    --distill-assignment \
    --distilled-model wordemb \
    --distill-factor 0.3 \
    --criterion xentropy_aux \
    --balance-loss balance \
    --balance-factor 0.3 \
    --capacity-factor 2 \
    --assignment-algorithm GA \
    --share-decoder-input-output-embed \
    --sample-break-mode none \
    --tokens-per-sample 128 \
    --optimizer adam --adam-betas "(0.9, 0.98)" \
    --adam-eps 1e-08 \
    --clip-norm 0.0 \
    --lr 5e-4 \
    --lr-scheduler polynomial_decay \
    --warmup-updates 750 \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --weight-decay 0.01 \
    --batch-size 4 \
    --update-freq 1 \
    --required-batch-size-multiple 1 \
    --total-num-update 50000 \
    --max-update 50000 \
    --seed 1 \
    --ddp-backend=no_c10d \
    --batch-size 8 \
    --pad-to-fixed-length \
    --pad-to-fixed-bsz \
    --update-freq 512 \
    --ddp-backend=no_c10d \
    --log-interval 100 \
    --validate-interval-updates 500 \
    --save-interval 5000 \
    --tensorboard-logdir ../tblogs/$jobname \
    --log-file ../checkpoints/$jobname/log.txt \
    --log-format tqdm 