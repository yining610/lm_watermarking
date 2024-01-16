import click
import os
import torch
import transformers
import json

from extended_watermark_processor import WatermarkDetector
from summarization.src.metrics.metric import Metric
from summarization.scripts.analysis import compute_avg, compute_confusion_matrix, compute_min_expected_green_tokens

__CACHE_DIR__ = os.environ.get("CACHE_DIR")

@click.command()
@click.option("--dataset-dir", type=click.Path(exists=True), help="Path to the dataset directory.")
@click.option("--model-name",  type=click.Choice(["gpt-4", "gpt-3.5-turbo", "t5-large", "gpt2"]), help="The model to use.")
@click.option("--min-spike-entropy", type=click.FLOAT, default=0.6385, help="The minimum spike entropy.)") # TODO: pass this number automatically

def main(dataset_dir,
         model_name,
         min_spike_entropy):
    """Detect the watermark in the text.
    """
    dataset = torch.load(os.path.join(dataset_dir, "eval_outputs.pt"))
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name,
                                                           cache_dir=__CACHE_DIR__)
    
    batch_text = tokenizer.batch_decode(dataset.int(), skip_special_tokens=True)
    watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                           delta=2.0,
                                           gamma=0.25,                # should match original setting
                                           seeding_scheme="selfhash", # should match original setting
                                           device="cuda:0",           # must match the original rng device type
                                           tokenizer=tokenizer,
                                           z_threshold=2.0,
                                           normalizers=[],
                                           ignore_repeated_ngrams=True)
    
    outputs = []
    for output_text in batch_text:
        score_dict = watermark_detector.detect(output_text)
        score_dict['z_score_at_T'] = score_dict['z_score_at_T'].tolist()
        outputs.append(score_dict)
    
    # perform analysis
    compute_avg(outputs)
    compute_confusion_matrix(outputs)
    compute_min_expected_green_tokens(watermark_detector.delta, 
                                      watermark_detector.gamma, 
                                      min_spike_entropy)

if __name__ == "__main__":
    main()