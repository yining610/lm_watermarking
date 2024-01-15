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

1. Prepare CNN generation task dataset: `python steps/prepare_dataset.py --task-name cnn_generation --split {SPLIT} --output-dir ./data/processed_data/cnn`
2. Watermark CNN generation and compute spike entropy: ``