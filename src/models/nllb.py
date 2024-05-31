import regex as re
from models.from_checkpoint import ModelFromCheckpoint


class NLLBCheckpointOutput(object):
    """
    Represents the output of a Next-Language Learning Benchmark model checkpoint.

    Attributes:
        example_quality_mask (torch.Tensor): Mask indicating the quality of examples in the batch.
        batch_output_text (list): List of output texts generated by the model for each example in the batch.
    """

    def __init__(self, batch_output_text):
        """
        Initializes the NLLBCheckpointOutput object.

        Args:
            batch_output_text (list): List of output texts generated by the model for each example in the batch.
        """
        super(NLLBCheckpointOutput, self).__init__()
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

    def forward(self, batch):
        batch.collate(tokenizer=self.tokenizer)
        batch_input_ids = batch.inputs.input_ids.to(self.custom_device)
        batch_input_attention_mask = batch.inputs.attention_mask.to(self.custom_device)

        print('self.generate_kwargs', self.generate_kwargs)
        output_ids = self.model.generate(
            input_ids=batch_input_ids,
            attention_mask=batch_input_attention_mask,
            **self.generate_kwargs,
        )

        print('got here')

        batch_output_text = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )

        return NLLBCheckpointOutput(
            batch_output_text=batch_output_text,
        )
