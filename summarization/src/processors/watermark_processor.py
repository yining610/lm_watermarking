from abc import ABC, abstractmethod
import torch
from transformers import LogitsProcessor
from extended_watermark_processor import WatermarkBase

    
class Seq2SeqWatermarkLogitsProcessor(WatermarkBase, LogitsProcessor):

    """Watermark processor for Seq2Seq model.
    Because seq2seq model.generate() returns decoder's input_ids, which doesn't work for 
    the special seed scheme with context width > 1
    """

    def __init__(self, *args, store_spike_ents: bool = False, **kwargs):
        super().__init__(*args, **kwargs)

        self.store_spike_ents = store_spike_ents
        self.encoder_input_ids = None
        self.spike_entropies = None
        if self.store_spike_ents:
            self._init_spike_entropies()

    def _init_spike_entropies(self):
        alpha = torch.exp(torch.tensor(self.delta)).item()
        gamma = self.gamma

        self.z_value = ((1 - gamma) * (alpha - 1)) / (1 - gamma + (alpha * gamma))
        self.expected_gl_coef = (gamma * alpha) / (1 - gamma + (alpha * gamma))

        # catch for overflow when bias is "infinite"
        if alpha == torch.inf:
            self.z_value = 1.0
            self.expected_gl_coef = 1.0

    def _get_spike_entropies(self):
        spike_ents = [[] for _ in range(len(self.spike_entropies))]
        for b_idx, ent_tensor_list in enumerate(self.spike_entropies):
            for ent_tensor in ent_tensor_list:
                spike_ents[b_idx].append(ent_tensor.item())
        return spike_ents

    def _get_and_clear_stored_spike_ents(self):
        spike_ents = self._get_spike_entropies()
        self.spike_entropies = None
        return spike_ents

    def _compute_spike_entropy(self, scores):
        # precomputed z value in init
        probs = scores.softmax(dim=-1)
        denoms = 1 + (self.z_value * probs)
        renormed_probs = probs / denoms
        sum_renormed_probs = renormed_probs.sum()
        return sum_renormed_probs

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        # Cannot lose loop, greenlists might have different lengths
        green_tokens_mask = torch.zeros_like(scores, dtype=torch.bool)
        for b_idx, greenlist in enumerate(greenlist_token_ids):
            if len(greenlist) > 0:
                green_tokens_mask[b_idx][greenlist] = True
        return green_tokens_mask

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def _get_encoder_input_ids(self, input_ids: torch.LongTensor):
        self.encoder_input_ids = input_ids
    
    def _hook_encoder_decoder_input_ids(self, input_ids: torch.LongTensor, b_idx: int) -> torch.LongTensor:
        """ Hook encoder_input_ids and decoder_input_ids together to generate seed for seq2seq watermarking
        """
        assert self.encoder_input_ids is not None, "encoder_input_ids is None. Cannot concatenate with decoder_input_ids to generate seed"
        return torch.cat([self.encoder_input_ids[b_idx], input_ids], dim=0)

    def _score_rejection_sampling(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, tail_rule="fixed_compute") -> list[int]:
        """Generate greenlist based on current candidate next token. Reject and move on if necessary. Method not batched.
        This is only a partial version of Alg.3 "Robust Private Watermarking", as it always assumes greedy sampling. It will still (kinda)
        work for all types of sampling, but less effectively.
        To work efficiently, this function can switch between a number of rules for handling the distribution tail.
        These are not exposed by default.
        """
        sorted_scores, greedy_predictions = scores.sort(dim=-1, descending=True)

        final_greenlist = []
        for idx, prediction_candidate in enumerate(greedy_predictions):
            greenlist_ids = self._get_greenlist_ids(torch.cat([input_ids, prediction_candidate[None]], dim=0))  # add candidate to prefix
            if prediction_candidate in greenlist_ids:  # test for consistency
                final_greenlist.append(prediction_candidate)

            # What follows below are optional early-stopping rules for efficiency
            if tail_rule == "fixed_score":
                if sorted_scores[0] - sorted_scores[idx + 1] > self.delta:
                    break
            elif tail_rule == "fixed_list_length":
                if len(final_greenlist) == 10:
                    break
            elif tail_rule == "fixed_compute":
                if idx == 40:
                    break
            else:
                pass  # do not break early
        return torch.as_tensor(final_greenlist, device=input_ids.device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Call with previous context as input_ids, and scores for next token."""

        # this is lazy to allow us to co-locate on the watermarked model's device
        self.rng = torch.Generator(device=input_ids.device) if self.rng is None else self.rng

        # NOTE, it would be nice to get rid of this batch loop, but currently,
        # the seed and partition operations are not tensor/vectorized, thus
        # each sequence in the batch needs to be treated separately.

        list_of_greenlist_ids = [None for _ in input_ids]  # Greenlists could differ in length
        for b_idx, input_seq in enumerate(input_ids):
            if self.self_salt:
                input_seq = self._hook_encoder_decoder_input_ids(input_seq, b_idx)
                greenlist_ids = self._score_rejection_sampling(input_seq, scores[b_idx])
            else:
                greenlist_ids = self._get_greenlist_ids(input_seq)
            list_of_greenlist_ids[b_idx] = greenlist_ids

            # logic for computing and storing spike entropies for analysis
            if self.store_spike_ents:
                if self.spike_entropies is None:
                    self.spike_entropies = [[] for _ in range(input_ids.shape[0])]
                self.spike_entropies[b_idx].append(self._compute_spike_entropy(scores[b_idx]))

        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=list_of_greenlist_ids)
        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta)

        return scores

