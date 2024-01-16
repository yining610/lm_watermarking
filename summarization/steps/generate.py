import click
import os
import torch
from torch.utils.data import DataLoader

from summarization.src.utils.configs import get_params
from summarization.src.trainers.trainer import Trainer

@click.command()
@click.option("--dataset-dir", type=click.Path(exists=True), help="Path to the dataset directory.")
@click.option("--model-name",  type=click.STRING, help="The model to use.")
@click.option("--train-sample", type=click.INT, default=5000, help="The number of samples to train.")
@click.option("--eval-sample", type=click.INT, default=500, help="The number of samples to evaluate.")
@click.option("--batch-size", type=click.INT, default=8, help="The batch size for generation.")
@click.option("--epochs", type=click.INT, default=10, help="The number of epochs to train.")
@click.option("--learning-rate", type=click.FLOAT, default=1e-4)
@click.option("--output-dir", type=click.Path(exists=False), help="The output directory.")
@click.option("--gamma", type=click.FLOAT, default=0.25, help="The gamma parameter controlling gree list ratio")
@click.option("--delta", type=click.FLOAT, default=2.0, help="The delta parameter smoothing the logits")
@click.option("--seeding-scheme", type=click.Choice(["selfhash", "simple_1"]), default="selfhash", help="The seeding scheme to use")
@click.option("--do-eval", is_flag=True, help="Whether to only evaluate the finetuned model")

def main(dataset_dir,
         model_name,
         train_sample,
         eval_sample,
         batch_size,
         epochs,
         learning_rate,
         output_dir,
         gamma,
         delta,
         seeding_scheme,
         do_eval):
    """Complete the text given a prompt.
    """
    
    params = get_params(task_name="generation",
                        dataset_dir=dataset_dir,
                        model_name=model_name,
                        train_sample=train_sample,
                        eval_sample=eval_sample,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        output_dir=output_dir,
                        gamma=gamma,
                        delta=delta,
                        seeding_scheme=seeding_scheme,
                        do_eval=do_eval)
    
    trainer: Trainer = params["trainer"]
    dataloader_train: DataLoader = params["dataloader_train"]
    dataloader_eval: DataLoader = params["dataloader_eval"]

    # evaluate
    eval_dict = trainer.evaluate(
        dataloader=dataloader_eval,
        epoch=0
    )

    print(eval_dict)

    if not do_eval:
        # train and evaluate. The evaluation results are stored in trainer.save_dir
        trainer.train(
            dataloader=dataloader_train,
            eval_dataloader=dataloader_eval,
            epochs=epochs,
            patience=3,
        )

    # store eval outputs for later watermarking detection
    torch.save(trainer.eval_outputs,
               os.path.join(trainer.save_dir, "eval_outputs.pt"))

if __name__ == "__main__":
    main()