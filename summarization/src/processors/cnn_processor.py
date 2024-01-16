"""A set of preprocessor for the preprocessing
of CNN dataset.
"""
from typing import Dict, Text, Any, Union
import os
import datasets
from overrides import overrides
from functools import partial
import transformers

from .processor import Preprocessor, PreprocessorOutput
from experiments.watermark import check_input_lengths

__CACHE_DIR__ = os.environ["CACHE_DIR"]
if __CACHE_DIR__ is None:
    raise ValueError("CACHE_DIR not set")

class CNNGenerationProcessor(Preprocessor):
    """Prepare CNN dataset for generation
    """

    def __init__(self,
                 hf_model_name: Text,
                 min_sample_len: int,
                 min_prompt_len: int,
                 max_new_tokens: int,
                 ):
        super().__init__(batched=False)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model_name, 
                                                                    cache_dir=__CACHE_DIR__)
        # input_truncation_strategy: completion_length
        self.token_kwargs = dict(
            hf_model_name=hf_model_name,
            completion_length=max_new_tokens,
        )

        # input_filtering_strategy: prompt_and_completion_length
        self.input_check_kwargs = dict(
            min_sample_len = min_sample_len,
            min_prompt_len = min_prompt_len,
            min_completion_len = max_new_tokens
        )

    @overrides
    def __call__(
        self, dataset: datasets.Dataset,
        **kwargs
    ) -> Union[PreprocessorOutput, datasets.Dataset]:
        dataset = dataset.map(
            partial(self._call, **self.token_kwargs),
            batched=self._batched,
            with_indices=False,
            **kwargs
        )

        dataset = dataset.filter(
            partial(check_input_lengths, **self.input_check_kwargs),
            batched=self._batched,
            with_indices=True
        )

        return dataset
    
    def _call(self, example: Dict[Text, Any], *args, **kwargs) -> Dict[Text, Any]:
        """adapted from 
        `experiments.watermark.tokenize_for_generation`
        to fit the processor interface
        """

        example = self.tokenize_and_truncate(example, 
                                             **kwargs)
        inputs = example["inputs"]
        # for calculating the baseline violation rate across the "gold" completion
        untruncated_inputs = example["untruncated_inputs"]

        # decode the preprocessed input to store for audit
        truncated_input = self.tokenizer.batch_decode(inputs, skip_special_tokens=True)[0]

        # also decode the original suffix of the input for audit as the baseline
        decoded_untruncated_input = self.tokenizer.batch_decode(untruncated_inputs, skip_special_tokens=True)[0]
        baseline_completion = decoded_untruncated_input.replace(truncated_input,"")

        return {
            "input_ids": inputs[0],
            "untruncated_input": untruncated_inputs[0],
            "truncated_input": truncated_input,
            "baseline_completion": baseline_completion,
            "orig_sample_length": untruncated_inputs.shape[1],
            "prompt_length": inputs.shape[1],
            "real_completion_length": untruncated_inputs.shape[1] - inputs.shape[1],
        }

    def tokenize_and_truncate(self,
                              example: dict,
                              completion_length: int = None,
                              prompt_length: int = None,
                              hf_model_name: str = None,
                              model_max_seq_len: int = 4096) -> Dict[Text, Any]:
        """take hf dataset entry and preprocess it for completion by a model"""
        assert hf_model_name is not None, "need model name to know whether to adjust wrt special tokens"
        assert "text" in example, "expects 'text' field to be present"
        # tokenize
        inputs = self.tokenizer.encode(example["text"], return_tensors="pt", truncation=True, max_length=model_max_seq_len)
        untruncated_inputs = inputs.clone()

        if (completion_length is not None) and (prompt_length is None):
            # leave at least one token as prefix # FIXME I think plus 1 since 0 is start tok
            slice_length = min(inputs.shape[1]-1, completion_length)
        elif (prompt_length is not None) and (completion_length is None):
            desired_comp_len = (inputs.shape[1]-1) - prompt_length
            slice_length = desired_comp_len if desired_comp_len > 0 else 0
        else:
            raise ValueError((f"Can only tokenize and truncate based on either the desired prompt length or desired completion length,",
                              f" but got completion_length:{completion_length},prompt_length:{prompt_length}"))

        # truncate
        inputs = inputs[:,:inputs.shape[1]-slice_length]
        # logic depending on special tokens for the model
        if "t5" in hf_model_name or "T0" in hf_model_name: 
            inputs[0,-1] = 1

        return {
            "inputs": inputs,
            "untruncated_inputs": untruncated_inputs
        }
    
class CNNSummarizationProcessor(Preprocessor):
    """Prepare CNN dataset for summarization
    """
    def __init__(self,
                 batch_size: int,
                 prompt: Text):
        super().__init__(batched=True)
        
        self.batch_size = batch_size
        self.prompt = prompt
        self.template = "{prompt} {text}"

    def _call(self, examples: Dict[Text, Any], *args, **kwargs) -> Dict[Text, Any]:

        templated_text = [
            self.template.format(prompt=self.prompt, text=article)
            for article in examples['text']
        ]
        
        return {
            "input_text": templated_text,
        }