CUDA_VISIBLE_DEVICES=0 fairseq-train --task truncated_bptt_lm  \
    /home/gtruong/Project/ICML/Fair_MOE/fairseq-moe/preprocess_scripts/data/enwik8/data-bin \
    --user-dir examples/adaptive_span \
    --fp16 --fp16-no-flatten-grads --max-update 600000 \
    --tokens-per-sample 512 --arch adaptive_span \
    --n-layer 12 --d-model 512 --n-head 8 --d-inner 2048 --dropout 0.3 \
    --attn-span 8192 --optimizer adagrad_with_grad_clip --adagrad-clip 0.03 \
    --validate-interval-updates 1000 \
    --lr-scheduler fixed --warmup-updates 32000 --batch-size-valid 32 \
    --lr 0.07 --criterion adaptive_span_loss --batch-size 16 --update-freq 1 \
    --seed 2 --log-format json --log-interval 25 --aux-loss-scaler 5e-07