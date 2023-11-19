fairseq-preprocess --only-source --trainpref data/enwik8/train.txt \
    --validpref data/enwik8/valid.txt --testpref data/enwik8/test.txt \
    --destdir data/enwik8/data-bin/ --joined-dictionary --workers 20