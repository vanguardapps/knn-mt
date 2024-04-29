import regex as re
from model.from_checkpoint import ModelFromCheckpoint


class NLLBCheckpointOutput(object):
    def __init__(self, example_quality_mask, batch_output_text):
        super(NLLBCheckpointOutput, self).__init__()
        self.example_quality_mask = example_quality_mask
        self.batch_output_text = batch_output_text


class NLLBCheckpoint(ModelFromCheckpoint):
    """Wrapper to load NLLB checkpoint model"""

    def __init__(
        self,
        checkpoint,
        src_lang,
        tgt_lang,
        **kwargs,
    ):
        """Initialize the NLLB model and tokenizer with a Hugging Face checkpoint.

        Args:
            checkpoint:
                Specifies the string checkpoint of the model to load. Either a HF
                hub checkpoint or the path to the local model checkpoint directory.
            src_lang:
                The BCP-47 code for the source language.
            tgt_lang:
                The BCP-47 code for the source language.
            **kwargs:
                Keyword arguments to be passed as necessary to the model, model config,
                and tokenizer initialization functions.
        """
        super(NLLBCheckpoint, self).__init__(
            checkpoint,
            **(
                kwargs
                | dict(
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    forced_bos_token_tgt_lang=tgt_lang,
                )
            ),
        )

    @staticmethod
    def _get_quality_mask(input_text):
        """Return a list of boolean values as a mask for per-sequence input quality.

        There are two criteria for a quality sequence. One is that the source text must contain
        a reasonable distribution of word characters spread across at least two words. This is
        achieved through several sequential regular expresion patterns, and is currently
        configured to test for at least 5 word characters shared between at minimum 2 words.
        Additionally, it is notable that using this logic, any sentences with only one word will
        automatically be rejected.

        The second criteria is the source text must contain less than 65% non-word characters.

        Args:
            input_text: Batch of string input sequences.
        """
        quality_mask = []

        non_word_threshold = 0.65

        patterns = [
            r'(\pL{1,}).+(\pL{4,})',
            r'(\pL{2,}).+(\pL{3,})',
            r'(\pL{3,}).+(\pL{2,})',
            r'(\pL{4,}).+(\pL{1,})',
        ]
        non_word = r'[^\pL]'

        for sentence in input_text:
            regex_results = [
                1 if bool(re.search(pattern, sentence)) else 0 for pattern in patterns
            ]
            if sum(regex_results) == 0:
                quality_mask.append(False)
                continue

            chars = list(sentence)
            punct_mask = [1 if bool(re.search(non_word, char)) else 0 for char in chars]

            if sum(punct_mask) >= non_word_threshold * len(chars):
                quality_mask.append(False)
                continue

            quality_mask.append(True)

        return quality_mask

    def _collate_fn(self, batch):
        batch_inputs = self.tokenizer(batch.inputs, padding=True, return_tensors='pt')
        return batch_inputs

    def forward(self, batch):
        batch_inputs = self._collate_fn(batch)
        batch_input_ids = batch_inputs.input_ids.to(self.custom_device)
        batch_input_attention_mask = batch_inputs.attention_mask.to(self.custom_device)

        output_ids = self.model.generate(
            input_ids=batch_input_ids, attention_mask=batch_input_attention_mask
        )

        batch_output_text = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )

        example_quality_mask = NLLBCheckpoint._get_quality_mask(input_text=batch.inputs)

        return NLLBCheckpointOutput(
            batch_output_text=batch_output_text,
            example_quality_mask=example_quality_mask,
        )
