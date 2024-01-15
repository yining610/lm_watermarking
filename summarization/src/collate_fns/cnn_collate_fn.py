from overrides import overrides
from typing import Dict, Text, Any, List
import torch
from transformers import PreTrainedTokenizer
from .collate_fn import CollateFn

class CNNGenerationCollateFn(CollateFn):

    def __init__(self,
                 tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    @overrides
    def collate(
        self,
        x: List[Dict[Text, Any]]
    ) -> Dict[Text, Any]:
    
        input_ids = {"input_ids": [example['input_ids'] for example in x]}
        padded_input_ids = self.tokenizer.pad(input_ids, 
                                              padding=True, 
                                              return_attention_mask=False,
                                              pad_to_multiple_of=8,
                                              return_tensors="pt")
        
        labels = [example['baseline_completion'] for example in x]
        labels = self.tokenizer(labels)['input_ids']
        return {
            **padded_input_ids,
            "labels": labels,
        }