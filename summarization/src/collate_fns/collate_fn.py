"""A processor that takes an input and construct it into a format
that will become the input of the model.

This processor is intended to be called inline with the training
script as it is usually lightweight and does not require a lot of
GPU computation.
"""
from typing import List, Dict, Tuple, Text, Any, Iterable
from abc import ABC, abstractmethod


class CollateFn(ABC):
    def __init__(
        self
    ):
        super().__init__()
        
    def __call__(self, x: List[Dict[Text, Any]]) -> Dict[Text, Any]:
        """
        """
        return self.collate(x)
    
    @abstractmethod
    def collate(self, x: List[Dict[Text, Any]]) -> Dict[Text, Any]:
        """
        """
        raise NotImplementedError("CollateFn is an abstract class.")