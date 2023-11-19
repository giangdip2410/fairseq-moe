PATH_TO_DATA="/home/gtruong/Project/ICML/Fair_MOE/fairseq-moe/preprocess_scripts/data/enwik8/data-bin"
jobname="smoe_transformer_lm_enk8_small"
root_path="/home/gtruong/Project/ICML/Fair_MOE/fairseq-moe/train_scripts/SMOE_transformer"
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
    --num-workers 8 \
    --activation-fn gelu \
    --share-decoder-input-output-embed \
    --validate-interval-updates 1000 \
    --save-interval-updates 1000 \
    --no-epoch-checkpoints \
    --memory-efficient-fp16 \
    --fp16-init-scale 4 \
    --arch transformer_lm_small \
    --sample-break-mode none \
    --save-dir $root_path/checkpoints/$jobname \
    --tokens-per-sample 128 \
    --optimizer adam --adam-betas "(0.9, 0.98)" \
    --adam-eps 1e-08 \
    --clip-norm 0.0 \
    --lr 0.0007 \
    --lr-scheduler polynomial_decay \
    --warmup-updates 2000 \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --weight-decay 0.01 \
    --batch-size 128 \
    --update-freq 16 \
    --required-batch-size-multiple 1 \
    --total-num-update 10000 \
    --max-update 10000 \
    --seed 2410 \
    --ddp-backend=no_c10d \
    --moe-expert-count 16 --moe-freq 1 \
    --moe-gating-use-fp32 --moe-second-expert-policy random --moe-normalize-gate-prob-before-dropping \
    --moe-eval-capacity-token-fraction -1.0 \
    --criterion moe_cross_entropy --moe-gate-loss-wt 0.01 --moe-gate-loss-combine-method sum \
    --tensorboard-logdir $root_path/tblogs/$jobname \
    --log-file $root_path/checkpoints/$jobname/log.txt \
    --log-format tqdm 