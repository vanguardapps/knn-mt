import numpy as np
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput
from models.nllb import NLLBCheckpoint


class NLLBEmbeddingsModel(NLLBCheckpoint):
    """Helper class to return autoregressive decoder embeddings from NLLB checkpoint."""

    def __init__(
        self,
        checkpoint,
        src_lang,
        tgt_lang,
        **kwargs,
    ):
        """Initialize the NLLB model and configure model to output hidden states.

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
        super(NLLBEmbeddingsModel, self).__init__(
            checkpoint,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            **(kwargs | dict(output_hidden_states=True)),
        )

        # Freeze base model (not necessary as the primary forward() operation is run using torch.no_grad())
        for param in self.model.base_model.parameters():
            param.requires_grad = False

    def forward(self, batch):
        batch.collate(tokenizer=self.tokenizer)

        batch_input_ids = batch.inputs.input_ids.to(self.custom_device)
        batch_input_attention_mask = batch.inputs.attention_mask.to(self.custom_device)
        batch_label_ids = batch.labels.input_ids.to(self.custom_device)

        batch_size = batch.inputs.input_ids.size(0)

        token_hidden_states = torch.empty(
            (batch_size, 0, self.model_config.hidden_size), dtype=torch.float32
        ).to(self.custom_device)

        token_hidden_states.requires_grad = False

        label_id_count = batch.labels.input_ids.size(1)

        self.model.eval()

        with torch.no_grad():
            for index in range(1, label_id_count + 1):
                # Forward pass model
                model_outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_input_attention_mask,
                    decoder_input_ids=batch_label_ids[:, :index].to(self.custom_device),
                )

                # Get all of the encoder representations on the first pass
                if index == 1:
                    encoder_last_hidden_state = (
                        # .detach() not necessary but illustrates intent (would still work without torch.no_grad())
                        model_outputs.encoder_last_hidden_state.detach()
                    )

                # Get index of token currently being decoded
                token_rep_idx = index - 1

                # Get decoder representation
                # .detach() not necessary but illustrates intent (would still work without torch.no_grad())
                last_hidden_state = model_outputs.decoder_hidden_states[-1].detach()
                token_representation = last_hidden_state[:, np.newaxis, token_rep_idx]

                # Concatenate decoder last hidden state of current token
                token_hidden_states = torch.cat(
                    (token_hidden_states, token_representation), dim=1
                )

        batch.encoder_last_hidden_state = encoder_last_hidden_state
        batch.token_hidden_states = token_hidden_states
        batch.inputs_tokenized = [
            self.tokenizer.convert_ids_to_tokens(input_ids)
            for input_ids in batch.inputs.input_ids
        ]
        batch.labels_tokenized = [
            self.tokenizer.convert_ids_to_tokens(input_ids)
            for input_ids in batch.labels.input_ids
        ]
