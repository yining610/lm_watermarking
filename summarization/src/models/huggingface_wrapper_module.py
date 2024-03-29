"""Huggingface model wrapper that has the
saving property.
"""
import os
import json
import transformers
from overrides import overrides
from .model import Model
CACHE_DIR="/scratch/ylu130/model-hf"

class HuggingfaceWrapperModule(Model):
    def __init__(
        self,
        model_handle: str,
    ):
        super().__init__()
        self.model_handle = model_handle
        if model_handle.startswith("t5"):
            self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                self.model_handle,
                cache_dir=CACHE_DIR
            )
        elif model_handle.startswith("gpt"):
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_handle,
                cache_dir=CACHE_DIR
            )
        else:
            raise NotImplementedError("Model is not supported.")

    def forward(self, *args, **kwargs):
        """Forward generation of the model.
        """
        model_outputs = self.model(*args, **kwargs)
        # append predictions to the outputs
        return {
            "loss": model_outputs.loss,
            "logits": model_outputs.logits,
            "predictions": model_outputs.logits.argmax(dim=-1)
        }
        
    def generate(self, *args, **kwargs):
        """Generate from the model.
        """
        return self.model.generate(*args, **kwargs)
    
    @overrides
    def save_to_dir(self, path: str):
        """Save the model to the given path.
        """
        self.model.save_pretrained(path)
        
        # save the model_handle as well.
        with open(os.path.join(path, 'custom.config'), 'w', encoding='utf-8') as file_:
            json.dump({
                'model_handle': self.model_handle
            }, file_, indent=4, sort_keys=True)
    
    @classmethod
    def load_from_dir(cls, path: str) -> "HuggingfaceWrapperModule":
        """Load the model from the given path.
        """
        with open(os.path.join(path, 'custom.config'), 'r', encoding='utf-8') as file_:
            config = json.load(file_)
            
        model_handle = config['model_handle']
        model = cls(model_handle=model_handle)
        # overwrite the model with the loaded one
        if "gpt" in model_handle:
            model.model = transformers.AutoModelForCausalLM.from_pretrained(path)
        elif "t5" in model_handle:
            model.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(path)
        else:
            raise NotImplementedError("Model is not supported.")
        return model