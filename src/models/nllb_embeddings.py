import numpy as np
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput
from model.nllb import NLLBCheckpoint


class NLLBEmbeddingsModelOutput(BaseModelOutput):
    def __init__(
        self,
        batch_inputs,
        batch_labels,
        encoder_last_hidden_state,
        token_hidden_states,
        token_ids,
    ):
        super(NLLBEmbeddingsModelOutput, self).__init__(
            last_hidden_state=token_hidden_states[-1]
        )
        self.batch_inputs = batch_inputs
        self.batch_labels = batch_labels
        self.encoder_last_hidden_state = encoder_last_hidden_state
        self.token_hidden_states = token_hidden_states
        self.token_ids = token_ids


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

    def _collate_fn(self, batch):
        batch_inputs = self.tokenizer(
            batch.inputs,
            padding=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )

        batch_labels = self.tokenizer(
            text_target=batch.labels,
            padding=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )

        return batch_inputs, batch_labels

    def forward(self, batch):
        batch_inputs, batch_labels = self._collate_fn(batch)

        batch_input_ids = batch_inputs.input_ids.to(self.custom_device)
        batch_input_attention_mask = batch_inputs.attention_mask.to(self.custom_device)
        batch_label_ids = batch_labels.input_ids.to(self.custom_device)

        batch_size = batch_inputs.input_ids.size(0)

        token_hidden_states = torch.empty(
            (batch_size, 0, self.model_config.hidden_size), dtype=torch.float32
        ).to(self.custom_device)
        token_ids = torch.empty((batch_size, 0), dtype=torch.int64).to(
            self.custom_device
        )

        label_id_count = batch_labels.input_ids.size(1)

        # TODO: use the below special_tokens_mask to mask out unneeded source encoder embeddings
        #       from the array, as well as to do the same for the decoded embeddings / and
        #       target token_ids

        # en_masked = ma.array(
        #         en_tokenized.input_ids, mask=en_tokenized.special_tokens_mask
        #     ).compressed()

        for index in range(1, label_id_count + 1):
            # Forward pass model
            model_outputs = self.model(
                input_ids=batch_input_ids,
                attention_mask=batch_input_attention_mask,
                decoder_input_ids=batch_label_ids[:, :index].to(self.custom_device),
            )

            # Get all of the encoder representations on the first pass
            if index == 1:
                encoder_last_hidden_state = model_outputs.encoder_last_hidden_state

            # Get index of token currently being decoded
            token_rep_idx = index - 1

            # Get decoder representation
            last_hidden_state = model_outputs.decoder_hidden_states[-1]
            token_representation = last_hidden_state[:, np.newaxis, token_rep_idx]

            # Concatenate decoder last hidden state of current token
            token_hidden_states = torch.cat(
                (token_hidden_states, token_representation), dim=1
            )

        return NLLBEmbeddingsModelOutput(
            batch_inputs=batch_inputs,
            batch_labels=batch_labels,
            encoder_last_hidden_state=encoder_last_hidden_state,
            token_hidden_states=token_hidden_states,
            token_ids=token_ids,
        )
