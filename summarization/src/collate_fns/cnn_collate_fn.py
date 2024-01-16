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
        
        labels = self.tokenizer(
            [
                example['baseline_completion'] for example in x
            ],
            padding="longest",
            return_tensors='pt'
        ).input_ids

        unpadded_input_ins = {"input_ids": [example['input_ids'] for example in x]}
        padded_input_ids = self.tokenizer.pad(unpadded_input_ins, 
                                              padding=True, 
                                              return_attention_mask=True,
                                              pad_to_multiple_of=8,
                                              return_tensors="pt")
        return {
            **padded_input_ids,
            "labels": labels,
        }
    
class CNNSummarizationCollateFn(CollateFn):

    def __init__(self,
                 tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
    
    @overrides
    def collate(
        self,
        x: List[Dict[Text, Any]]
    ) -> Dict[Text, Any]:
        
        labels = self.tokenizer(
            [
                example['highlights'] for example in x
            ],
            padding="longest",
            return_tensors='pt'
        ).input_ids

        inputs = [example['input_text'] for example in x]
        tokenized_inputs = self.tokenizer(inputs, 
                                          return_tensors='pt',
                                          padding=True)
        
        return {
            **tokenized_inputs,
            "labels": labels,
        }