"""This module contains the configurations for getting
inputs for the models.
"""

import datasets
import os
from typing import Text, Dict, Any, List, Tuple, Optional
import transformers
import torch
from torch.utils.data import DataLoader

from summarization.src.processors.watermark_processor import Seq2SeqWatermarkLogitsProcessor
from summarization.src.metrics.rouge import GenerationRouge
from summarization.src.metrics.spike_entropy import AvgSipkeEntropy
from summarization.src.models.huggingface_wrapper_module import HuggingfaceWrapperModule
from summarization.src.collate_fns.cnn_collate_fn import CNNGenerationCollateFn
from summarization.src.trainers.cnn_trainer import CNNTrainer

__CACHE_DIR__ = os.environ["CACHE_DIR"]
if __CACHE_DIR__ is None:
    raise ValueError("CACHE_DIR not set")

def get_generation_params(
    dataset_dir: Text,
    model_name: Text,
    num_sample: int,
    batch_size: int = 64,
    use_raw_model: bool = True
):
    
    if "cnn" in dataset_dir:
        task_name = "cnn"
        dataset = datasets.load_from_disk(os.path.join(dataset_dir, "validation"))
        # select first num_sample samples
        dataset = dataset.select(range(num_sample))
    else:
        raise ValueError("Dataset currently not supported.")
    
    if use_raw_model:
        model = HuggingfaceWrapperModule(model_handle=model_name)
    else:
        model = HuggingfaceWrapperModule.load_from_dir(
            f"{__CACHE_DIR__}/generation/{task_name}_{model_name}/best_1"
        )
        
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, 
                                                           cache_dir=__CACHE_DIR__)

    collate_fn = CNNGenerationCollateFn(tokenizer=tokenizer)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    watermark_processor = Seq2SeqWatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                          gamma=0.25,
                                                          delta=2.0,
                                                          store_spike_ents=True,
                                                          seeding_scheme="selfhash")

    trainer = CNNTrainer(
        model=model,
        logits_processor = watermark_processor,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.0),
        device="cuda:0",
        metrics={
            "rouge": GenerationRouge(tokenizer),
        },
        eval_metrics={
            "spike_entropy": AvgSipkeEntropy(),
            "rouge": GenerationRouge(tokenizer),
        },
        main_metric="rouge",
        save_dir=None,
    )

    return {
        "dataloader": dataloader,
        "trainer": trainer,
    }

