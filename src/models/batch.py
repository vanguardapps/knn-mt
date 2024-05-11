import numpy as np
import os
import regex as re
import subprocess
import torch


"""Batching Module

Batching is approached from an end-to-end standpoint by task. A batch is expected to contain
attributes for both the input and the output of the task, and in this way reduces memory
footprint by never requiring any data duplication. This also increases usability, as data
that belongs together remains together, side-by-side in the batch.
"""


class LMBatch(object):
    """
    Represents a generic Language Model batch.

    Attributes:
        inputs (list): List of input sequences.
        inputs_raw (list): List of raw input sequences.
    """

    supported_params = ["inputs", "inputs_raw"]

    def __init__(self, **kwargs):
        """
        Initializes the LMBatch object with the provided keyword arguments.

        Args:
            **kwargs: Arbitrary keyword arguments. Supported arguments are listed in `supported_params`.
        """
        for param_name in LMBatch.supported_params:
            value = kwargs.get(param_name, None)
            if value is not None:
                setattr(self, param_name, value)

    def collate(self, tokenizer):
        """
        Collates batch of input sequences using the provided tokenizer.

        This method prepares the input sequences for model processing by tokenizing
        and padding them using the given tokenizer.

        Args:
            tokenizer: A tokenizer object used to tokenize the input sequences.

        Note:
            This method assumes that `inputs_raw` attribute contains raw input sequences
            to be tokenized and padded. After collation, the resulting sequences are stored
            in the `inputs` attribute of the LMBatch object.
        """
        self.inputs = tokenizer(self.inputs_raw, padding=True, return_tensors="pt")

    def get_quality_mask(self):
        """Return a list of boolean values as a mask for per-sequence input quality.

        There are two criteria for a quality sequence. One is that the source text must contain
        a reasonable distribution of word characters spread across at least two words. This is
        achieved through several sequential regular expresion patterns, and is currently
        configured to test for at least 5 word characters shared between at minimum 2 words.
        Additionally, it is notable that using this logic, any sentences with only one word will
        automatically be rejected.

        The second criteria is the source text must contain less than 65% non-word characters.

        Note:
            This method assumes that `inputs_raw` attribute contains raw input sequences
            to be graded for quality.
        """
        quality_mask = []

        non_word_threshold = 0.65

        patterns = [
            r"(\pL{1,}).+(\pL{4,})",
            r"(\pL{2,}).+(\pL{3,})",
            r"(\pL{3,}).+(\pL{2,})",
            r"(\pL{4,}).+(\pL{1,})",
        ]
        non_word = r"[^\pL]"

        for sentence in self.inputs_raw:
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

    def postprocess(self):
        """Postprocess the batch after running through whatever model.

        This will be different for every kind of batch. In a lot of cases
        it will house logic to deal with the values that are expected
        back from the model used for the given task.
        """
        raise NotImplementedError("'postprocess' method not implemented in base class.")


class EmbeddingsBatch(LMBatch):
    """
    Represents a batch for the task of getting last hidden states per input token from
    an encoder / decoder checkpoint from Hugging Face, inheriting from LMBatch.

    Attributes:
        alignments (dict): Source token index to target token index alignments.
        inputs (list): List of input sequences.
        inputs_raw (list): List of raw input sequences.
        labels (list): List of target sequences.
        labels_raw (list): List of raw target sequences.
        encoder_last_hidden_state (list): Last hidden states of the encoder.
        token_hidden_states (list): Hidden states of each token.

    Note: labels are required for this task to be completed, as the last hidden states
    are intended to be retrieved from translation bitext where both source text and
    gold targets are known.
    """

    def __init__(self, **kwargs):
        """Initializes the EmbeddingsBatch object with the provided keyword arguments.

        Extends the supported_params attribute to include additional parameters required for translation tasks.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        EmbeddingsBatch.supported_params.extend(
            [
                "labels",
                "labels_raw",
                "encoder_last_hidden_state",
                "target_hidden_states",
            ]
        )
        super(EmbeddingsBatch, self).__init__(**kwargs)

    def collate(self, tokenizer):
        """Collates input and target sequences using a tokenizer.

        Args:
            tokenizer: Tokenizer object for processing input and target sequences.
        """
        self.inputs = tokenizer(
            self.inputs_raw,
            padding=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )

        self.labels = tokenizer(
            text_target=self.labels_raw,
            padding=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )

    def postprocess(self):
        """Generates postprocessed version of batch data with special character tokens masked out.

        Given:
            l = source sequence length
            m = target sequence length
            e = model embedding dimension

        Creates the following attributes on the class (m = particular sequence length):
            self.input_ids_masked                   (list[LongTensor(l)])
            self.label_ids_masked                   (list[LongTensor(m)])
            self.encoder_last_hidden_state_masked   (list[FloatTensor(m, e)])
            self.target_hidden_states_masked        (list[FloatTensor(m, e)])

        Note: It is expected that the batch has already been processed through a model and that the
        values for self.inputs, self.labels, self.encoder_last_hidden_state, and
        self.target_hidden_states are all available.
        """

        self.input_ids_masked = []
        self.label_ids_masked = []
        self.encoder_last_hidden_state_masked = []
        self.target_hidden_states_masked = []

        for (
            source_ids,
            source_mask,
            target_ids,
            target_mask,
            encoder_hidden_state,
            target_hidden_state,
        ) in zip(
            self.inputs.input_ids,
            self.inputs.special_tokens_mask,
            self.labels.input_ids,
            self.labels.special_tokens_mask,
            self.encoder_last_hidden_state,
            self.target_hidden_states,
        ):
            source_mask = np.invert(np.array(source_mask, dtype=bool))
            target_mask = np.invert(np.array(target_mask, dtype=bool))
            self.input_ids_masked.append(torch.LongTensor(source_ids[source_mask]))
            self.label_ids_masked.append(torch.LongTensor(target_ids[target_mask]))
            self.encoder_last_hidden_state_masked.append(
                torch.FloatTensor(encoder_hidden_state[source_mask])
            )
            self.target_hidden_states_masked.append(
                torch.FloatTensor(target_hidden_state[target_mask])
            )

    def generate_alignments(self, tokenizer, temp_filepath1=None, temp_filepath2=None):
        input_corpus_filepath = (
            temp_filepath1
            if temp_filepath1 is not None
            else "fast_align_temp_file1.tmp"
        )
        output_alignments_filepath = (
            temp_filepath2
            if temp_filepath2 is not None
            else "fast_align_temp_file2.tmp"
        )

        with open(input_corpus_filepath, "w") as corpus_file:
            for source_ids, target_ids in zip(
                self.input_ids_masked, self.label_ids_masked
            ):
                line = (
                    " ".join(tokenizer.convert_ids_to_tokens(source_ids))
                    + " ||| "
                    + " ".join(tokenizer.convert_ids_to_tokens(target_ids))
                    + "\n"
                )

                corpus_file.write(line)

        subprocess.run(
            f"./bin/fast_align -i {input_corpus_filepath} -d -o -v > {output_alignments_filepath}",
            shell=True,
        )

        with open(output_alignments_filepath) as alignments_file:
            alignments = []
            for line in alignments_file:
                pairs = re.split(r"\s+", line)
                line_alignments = {}

                for pair in pairs:
                    if not pair:
                        continue

                    split_pair = pair.split("-")
                    source_index = split_pair[0]
                    target_index = split_pair[1]

                    # Accept first alignment only for given source token
                    if not line_alignments.get(int(source_index), None):
                        line_alignments[int(source_index)] = int(target_index)

                alignments.append(line_alignments)

            self.alignments = alignments

        os.remove(input_corpus_filepath)
        os.remove(output_alignments_filepath)
