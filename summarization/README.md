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
1. Prepare CNN task dataset: `python steps/prepare_dataset.py --task-name --split --output-dir`
2. Train and evaluate generation model: `python steps/generate.py --dataset-dir --model-name`
3. Train and evaluate summarization model: `python steps/summarize.py --dataset-dir --model-name`
4. Detect watermark and analysis: `python scripts/watermark_detect.py --dataset-dir --model-name`
   
**Note:** Please refer to `run_all_generation.sh` and `run_all_summarization.sh` to see the complete running commands.