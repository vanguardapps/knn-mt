import regex as re

"""Batching Module

Batching is approached from an end-to-end standpoint by task. A batch is expected to contain
attributes for both the input and the output of the task, and in this way reduces memory
footprint by never requiring any data duplication. This also increases usability, as data
that belongs together remains together, side-by-side in the batch.
"""


def collate(batch, tokenizer):
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
    return tokenizer(batch.inputs_raw, padding=True, return_tensors="pt")


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


class EmbeddingsBatch(LMBatch):
    """
    Represents a batch for the task of getting last hidden states per input token from
    an encoder / decoder checkpoint from Hugging Face, inheriting from LMBatch.

    Attributes:
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
                "inputs_tokenized",
                "labels",
                "labels_raw",
                "labels_tokenized",
                "encoder_last_hidden_state",
                "token_hidden_states",
            ]
        )
        super(EmbeddingsBatch, self).__init__(**kwargs)

    def collate(self, tokenizer):
        """
        Collates input and target sequences using a tokenizer.

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
