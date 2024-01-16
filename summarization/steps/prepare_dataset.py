"""This script takes a step to convert data 
into a dataset that will be directly feeding to the model with
appropriate prompts and labels based on the task.
"""
import click
import os
import datasets

from summarization.src.processors.cnn_processor import (
    CNNGenerationProcessor,
    CNNSummarizationProcessor
)
__REGISTRY__ = {
    "cnn_generation": [
        {   
            "path": "cnn_dailymail",
            "cls": CNNGenerationProcessor,
            "params": {
                "hf_model_name": "t5-large",
                "min_sample_len": 0,      # default value same as in the original code
                "min_prompt_len": 50,     # default value same as in the original code
                "max_new_tokens": 200     # default value same as in the original code
            }
        }
    ],
    "cnn_summarization": [
        {
            "path": "cnn_dailymail",
            "cls": CNNSummarizationProcessor,
            "params": {
                "batch_size": 64,
                "prompt": "Summarize the following article: "
            }
        }
    ]
}

@click.command()
@click.option("--task-name", type=str, required=True, help="Task that the dataset will be used for.")
@click.option("--split", type=click.Choice(["train", "validation", "test"]), help="Which split to use.", default='validation')
@click.option("--output-dir",  type=click.Path(exists=False, file_okay=True), help="Path to write the output to (will be subdir-ed by data_name (not none) and split).")

def main(task_name,
         split,
         output_dir):
    
    dataset_path = __REGISTRY__[task_name][0]["path"]
    if dataset_path == "cnn_dailymail":
        dataset = datasets.load_dataset(dataset_path,
                                        "3.0.0",
                                        split=split,
                                        cache_dir=os.environ["CACHE_DIR_DATA"])
        # be compatible with the original code
        dataset = dataset.rename_column("article", "text")
    else:
        raise ValueError(f"Dataset {dataset_path} currently not supported.")
    
    for preprocessor_config in __REGISTRY__[task_name]:
        preprocessor = preprocessor_config["cls"](**preprocessor_config["params"])
        dataset = preprocessor(dataset)

    os.makedirs(output_dir, exist_ok=True)
    
    dataset.save_to_disk(os.path.join(output_dir, split))

if __name__ == "__main__":
    main()
