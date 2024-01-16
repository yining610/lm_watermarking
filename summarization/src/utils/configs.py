"""This module contains the configurations for getting
inputs for the models.
"""

import datasets
import os
from typing import Text, Dict, Any, List, Tuple, Optional
import transformers
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import wandb

from summarization.src.processors.watermark_processor import Seq2SeqWatermarkLogitsProcessor
from summarization.src.metrics.loss import AvgLoss
from summarization.src.metrics.rouge import GenerationRouge
from summarization.src.metrics.spike_entropy import AvgSipkeEntropy
from summarization.src.models.huggingface_wrapper_module import HuggingfaceWrapperModule
from summarization.src.collate_fns.cnn_collate_fn import (
    CNNGenerationCollateFn,
    CNNSummarizationCollateFn
)
from summarization.src.trainers.cnn_trainer import T5CNNTrainer

__CACHE_DIR__ = os.environ["CACHE_DIR"]
if __CACHE_DIR__ is None:
    raise ValueError("CACHE_DIR not set")

def get_params(
    task_name: Text,
    dataset_dir: Text,
    model_name: Text,
    train_sample: int,
    eval_sample: int,
    batch_size: int,
    learning_rate: bool,
    output_dir: Text,
    gamma: float = 0.25,
    delta: float = 2.0,
    seeding_scheme: Text = "selfhash",
    do_eval: bool = False,
):
    
    if "cnn" in dataset_dir:
        data_name = "cnn"
        dataset_train = datasets.load_from_disk(
            os.path.join(dataset_dir, "train")
            ).select(range(train_sample))
        dataset_eval = datasets.load_from_disk(
            os.path.join(dataset_dir, "validation")
            ).select(range(eval_sample))
    else:
        raise ValueError("Dataset currently not supported.")
    
    if not do_eval:
        wandb.init(project="watermark", 
                   name=f"{data_name}_{model_name}_{task_name}")
        model = HuggingfaceWrapperModule(model_handle=model_name)
    else:
        model = HuggingfaceWrapperModule.load_from_dir(
                    f"{output_dir}/{task_name}/{data_name}_{model_name}/best_1"
            )
        
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, 
                                                           cache_dir=__CACHE_DIR__)
    
    if task_name == "generation":
        collate_fn = CNNGenerationCollateFn(tokenizer=tokenizer)
    elif task_name == "summarization":
        collate_fn = CNNSummarizationCollateFn(tokenizer=tokenizer)
    else:
        raise ValueError("Task currently not supported.")

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    dataloader_eval = DataLoader(
        dataset_eval,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    watermark_processor = Seq2SeqWatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                          gamma=gamma,
                                                          delta=gamma,
                                                          store_spike_ents=True,
                                                          seeding_scheme=seeding_scheme)
    
    trainer = T5CNNTrainer(
        model=model,
        logits_processor=watermark_processor,
        optimizer=AdamW(
            params=model.parameters(),
            lr=learning_rate,
        ),
        metrics={
            "rouge": GenerationRouge(tokenizer),
            "loss": AvgLoss(),            
        },
        eval_metrics={
            "spike_entropy": AvgSipkeEntropy(),
            "rouge": GenerationRouge(tokenizer),
        },
        main_metric="rouge",
        direction='-',
        save_top_k=1,
        device="cuda:0",
        save_dir=f"{output_dir}/{task_name}/{data_name}_{model_name}",
        use_wandb=True,
    )

    return {
        "dataloader_train": dataloader_train,
        "dataloader_eval": dataloader_eval,
        "trainer": trainer,
    }
