PATH_TO_DATA="/home/gtruong/Project/ICML/Fair_MOE/fairseq-moe/preprocess_scripts/data-bin/wikitext-2"
jobname="stable_brainformer_lm_wt2_medium"
root_path="/home/gtruong/Project/ICML/Fair_MOE/fairseq-moe/train_scripts/StableMOE"
DIR=$root_path/checkpoints/$jobname
if [ ! -d "$DIR" ];
then
	mkdir $DIR
else
	echo "$DIR directory exists."
fi
filename=$root_path/checkpoints/$jobname/log.txt
if [[ ! -e filename ]]; then
    touch filename
fi
fairseq-train --task language_modeling \
    ${PATH_TO_DATA} \
    --num-workers 4 \
    --activation-fn gelu \
    --share-decoder-input-output-embed \
    --validate-interval-updates 1000 \
    --save-interval-updates 1000 \
    --no-epoch-checkpoints \
    --memory-efficient-fp16 \
    --fp16-init-scale 4 \
    --user-dir stablemoe \
    --arch brainformer_lm_stable_small \
    --moe-type base_layer \
    --two-stage-updates 3000 \
    --distill-assignment \
    --distilled-model wordemb \
    --distill-factor 0.3 \
    --criterion xentropy_aux \
    --balance-loss balance \
    --balance-factor 0.3 \
    --capacity-factor 2 \
    --assignment-algorithm GA \
    --sample-break-mode none \
    --save-dir ../checkpoints/$jobname \
    --tokens-per-sample 128 \
    --optimizer adam --adam-betas "(0.9, 0.98)" \
    --adam-eps 1e-08 \
    --clip-norm 0.0 \
    --lr 0.0007 \
    --lr-scheduler polynomial_decay \
    --warmup-updates 1000 \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --weight-decay 0.01 \
    --batch-size 64 \
    --update-freq 16 \
    --required-batch-size-multiple 1 \
    --total-num-update 10000 \
    --max-update 10000 \
    --seed 2410 \
    --ddp-backend=no_c10d \
    --tensorboard-logdir $root_path/tblogs/$jobname \
    --log-file $root_path/checkpoints/$jobname/log.txt \
    --log-format tqdm 