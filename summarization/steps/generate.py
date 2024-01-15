import click
import os
from torch.utils.data import DataLoader

from summarization.src.utils.configs import get_generation_params
from summarization.src.trainers.trainer import Trainer


@click.command()
@click.option("--dataset-dir", type=click.Path(exists=True), help="Path to the dataset directory.")
@click.option("--model-name",  type=click.Choice(["gpt-4", "gpt-3.5-turbo", "t5-large", "gpt2"]), help="The model to use.")
@click.option("--num-sample", type=click.INT, default=-1, help="The number of samples to generate.")
@click.option("--batch-size", type=click.INT, default=64, help="The batch size for generation.")
@click.option("--use-raw-model", is_flag=True, default=False, help="Use the raw model without finetuning.")

def main(dataset_dir,
         model_name,
         num_sample,
         batch_size,
         use_raw_model):
    """Complete the text given a prompt.
    """
    
    params = get_generation_params(dataset_dir=dataset_dir,
                                   model_name=model_name,
                                   num_sample=num_sample,
                                   batch_size=batch_size,
                                   use_raw_model=use_raw_model)
    
    trainer: Trainer = params["trainer"]
    dataloader: DataLoader = params["dataloader"]

    eval_dict = trainer.evaluate(
        dataloader=dataloader,
        epoch=0
    )
        
    print(eval_dict)

if __name__ == "__main__":
    main()