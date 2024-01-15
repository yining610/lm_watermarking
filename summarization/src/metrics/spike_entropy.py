"""calculate average spike entropy function.
"""
import torch
import numpy as np
from overrides import overrides
from .metric import Metric

class AvgSipkeEntropy(Metric):
    """Calculate the average spike entropy entropy of a generated sequence.
    """
    def __init__(self):
        """
        """
        super().__init__(name="avg_spike_entropy")
        self.entropy = []
        
    @overrides
    def __call__(self, outputs):
        """
        """
        outputs = self._detach_tensors(outputs)
        self.entropy.extend(outputs['spike_entropies'])
        
    @overrides
    def reset(self):
        """
        """
        self.loss = []
        
    @overrides
    def compute(self) -> float:
        """
        """

        return_ent = np.array(self.entropy).mean().item()
        self.reset()
        
        return return_ent