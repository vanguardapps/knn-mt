import torch
from transformers.generation import LogitsProcessor


class CustomLogitsProcessor(LogitsProcessor):
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, current_tokens
    ) -> torch.FloatTensor:
        print(current_tokens)
        return scores
