# Watermark on Summarization Task
Unless stated, all files discussed here are in the `summarization` directory: `cd summarization`

## Requirements
`pip install -r requirements.txt`
`echo "export CACHE_DIR=your-huggingface-model-cache-dir" >> ~/.bashrc`
`echo "export CACHE_DIR_DATA=your-huggingface-data-cache-dir" >> ~/.bashrc`

## File Structure Description
```shellscript
steps/   // callable scripts corresponds to each step of watermark detection.
src/     // source code of models, trainers, data collations etc. 
scripts/ // helper scripts to do examination, sanity check, analysis etc.
```

## Experiments
1. Prepare CNN task dataset: `python steps/prepare_dataset.py --task-name {TASK_NAME} --split {SPLIT} --output-dir {OUTPUT_DIR}`
2. Train and evaluate for generation task: `python steps/generate.py --dataset-dir {OUTPUT_DIR} --model-name {MODEL_NAME}`
3. Train and evaluate for summarization task: `python steps/summarize.py --dataset-dir {OUTPUT_DIR} --model-name {MODEL_NAME}`

4. Watermark CNN generation and compute spike entropy: `python steps/generate.py --dataset-dir ./data/processed_data/cnn_generation --model-name t5-large --num-sample 500 --output-dir ./data/cnn/generation --use-raw-model`
5. Watermark CNN summarization and compute spike entropy: `python steps/summarize.py --dataset-dir ./data/processed_data/cnn_summarization --model-name t5-large --num-sample 500 --output-dir ./data/cnn/summarization --use-raw-model`
6. Detect watermark for generation and analysis: `python scripts/watermark_detect.py --dataset-dir ./data/cnn/generation --model-name t5-large --output-dir ./data/cnn/generation --overwrite`
7. Detect watermark for summarization and analysis: `python scripts/watermark_detect.py --dataset-dir ./data/cnn/summarization --model-name t5-large --output-dir ./data/cnn/summarization --overwrite`