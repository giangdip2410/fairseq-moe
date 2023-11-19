NUM_EXPERTS=8
TOKENS_PER_SAMPLE=2048
fairseq-train --task language_modeling \
  /home/gtruong/Project/ICML/Fair_MOE/fairseq-moe/preprocess_scripts/data/enwik8/data-bin \
  --save-dir checkpoints/transformer_moe_enwik8 \
  --arch transformer_lm --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 2048 --update-freq 16 \
  --fp16-no-flatten-grads \
  --moe-expert-count $NUM_EXPERTS --moe-freq 2 \
  --moe-gating-use-fp32 --moe-second-expert-policy all \
  --moe-normalize-expert-grad sqrt_world_size \
  --moe-eval-capacity-token-fraction -1.0 \
  --max-sentences-valid 1 --num-workers-valid 0 \
  --criterion moe_cross_entropy --moe-gate-loss-wt 0.01 --moe-gate-loss-combine-method sum \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr 0.0005 --warmup-updates 750 \
  --dropout 0.1 --attention-dropout 0.1 \
  --batch-size 4 --update-freq 1 \
  --max-update 250 --disable-validation \
  --log-format json --log-interval 10 \
  --max-update 50000

# NUM_EXPERTS=8
# TOKENS_PER_SAMPLE=2048
# python fairseq_cli/train.py \
#   --ddp-backend fully_sharded --memory-efficient-fp16 --checkpoint-activations \
#   --task dummy_lm --tokens-per-sample $TOKENS_PER_SAMPLE \
#   --arch transformer_lm_gpt --share-decoder-input-output-embed \
#   --decoder-layers 24 --decoder-embed-dim 2048 --decoder-ffn-embed-dim 8192 \
#   --decoder-attention-heads 32 \
#   --moe-expert-count $NUM_EXPERTS --moe-freq 2 \
#   --moe-gating-use-fp32 --moe-second-expert-policy all \
#   --moe-normalize-expert-grad sqrt_world_size \
#   --moe-eval-capacity-token-fraction -1.0 \
#   --max-sentences-valid 1 --num-workers-valid 0 \
#   --criterion moe_cross_entropy --moe-gate-loss-wt 0.01 --moe-gate-loss-combine-method sum \
#   --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#   --lr 0.0005 --warmup-updates 750 \
#   --dropout 0.1 --attention-dropout 0.1 \
#   --batch-size 4 --update-freq 1 \
#   --max-update 250 --disable-validation \
#   --log-format json --log-interval 10