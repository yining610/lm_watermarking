# prepare dataset
for split in train validation test
    python steps/prepare_dataset.py \
            --task-name cnn_summarization \
            --split $split \
            --output-dir /scratch/ylu130/project/watermark/data/processed_data/cnn_summarization

# train and evaluate (including prior evaluation)
python steps/summarize.py \
        --dataset-dir /scratch/ylu130/project/watermark/data/processed_data/cnn_summarization \
        --model-name t5-base \
        --train-sample 5000 \
        --eval-sample 500 \
        --batch-size 2 \
        --epochs 10 \
        --learning-rate 1e-4 \
        --output-dir /scratch/ylu130/project/watermark/ckpt \
        --gamma 0.25 \
        --delta 2.0 \
        --seeding-scheme selfhash

# evaluation only (for scaling up experiments）
# python steps/summarize.py \
#         --dataset-dir /scratch/ylu130/project/watermark/data/processed_data/cnn_summarization \
#         --model-name t5-base \
#         --train-sample 5000 \
#         --eval-sample 500 \
#         --batch-size 32 \
#         --epochs 10 \
#         --learning-rate 1e-4 \
#         --output-dir /scratch/ylu130/project/watermark/ckpt \
#         --gamma 0.5 \
#         --delta 2.0 \
#         --seeding-scheme selfhash \
#         --do-eval

# watermark detection and analysis (need manually change the delta and gamma passing to the detector)
python scripts/watermark_detect.py \
        --dataset-dir /scratch/ylu130/project/watermark/ckpt/summarization/cnn_t5-base/ \
        --model-name t5-base