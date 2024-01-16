"""Overload the trainer eval_step to also evaluate results with the labels.
"""
from typing import Dict, Text, Any, Optional
import torch
from overrides import overrides
from .trainer import Trainer
from ..metrics.metric import Metric
from transformers import LogitsProcessor, LogitsProcessorList

class T5CNNTrainer(Trainer):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Any,
        metrics: Dict[Text, Any],
        eval_metrics: Dict[Text, Metric],
        main_metric: Text,
        save_dir: Text,
        device: Text,
        direction: Optional[Text] = '-',
        save_top_k: Optional[int] = 1,
        use_wandb: Optional[bool] = False,
        logits_processor: Optional[LogitsProcessor] = None,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            metrics=metrics,
            eval_metrics=eval_metrics,
            main_metric=main_metric,
            save_dir=save_dir, 
            device=device,
            direction=direction,
            save_top_k=save_top_k,
            use_wandb=use_wandb,
        )
        self.logits_processor = logits_processor
        
    @overrides
    def _train_step(self, batch: Dict[Text, Any]) -> Dict[Text, Any]:
        model_outputs = self.model(**batch)
        return {
            **model_outputs,
            "labels": batch["labels"],
        }
    
    @overrides
    def _eval_step(self, batch: Dict[Text, Any]) -> Dict[Text, Any]:

        labels = batch.pop("labels")
        
        self.logits_processor._get_encoder_input_ids(batch["input_ids"])
        model_outputs = self.model.generate(**batch,
                                            do_sample=True, 
                                            logits_processor=LogitsProcessorList([self.logits_processor]))
        
        spike_entropies = torch.tensor(self.logits_processor._get_and_clear_stored_spike_ents())

        return {
            "predictions": model_outputs,
            "labels": labels,
            "spike_entropies": spike_entropies,
        }
    