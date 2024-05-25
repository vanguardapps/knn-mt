import torch
from dotenv import load_dotenv
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from utils import dict_subset, validate_required_params

"""Hugging Face parameter lists.

Add parameters herein if using specialized models / tokenizers / model configs
with parameters not already listed (but allowed by the underlying entity).
"""
HF_GENERATE_FUNCTION_PARAMS = ["logits_processor"]
HF_MODEL_FROM_PRETRAINED_PARAMS = [
    "pretrained_model_name_or_path",
    "model_args",
    "config",
    "state_dict",
    "cache_dir",
    "from_tf",
    "force_download",
    "resume_download",
    "proxies",
    "output_loading_info(bool,",
    "local_files_only(bool,",
    "revision",
    "trust_remote_code",
    "code_revision",
    "token",
]
HF_MODEL_CONFIG_PARAMS = [
    "pretrained_model_name_or_path",
    "cache_dir",
    "force_download",
    "resume_download",
    "proxies",
    "revision",
    "return_unused_kwargs",
    "trust_remote_code",
    "name_or_path",
    "output_hidden_states",
    "output_attentions",
    "return_dict",
    "is_encoder_decoder",
    "is_decoder",
    "cross_attention_hidden_size",
    "add_cross_attention",
    "tie_encoder_decoder",
    "prune_heads",
    "chunk_size_feed_forward",
    "max_length",
    "min_length",
    "do_sample",
    "early_stopping",
    "num_beams",
    "num_beam_groups",
    "diversity_penalty",
    "temperature",
    "top_k",
    "top_p",
    "typical_p",
    "repetition_penalty",
    "length_penalty",
    "no_repeat_ngram_size",
    "encoder_no_repeat_ngram_size",
    "bad_words_ids",
    "num_return_sequences",
    "output_scores",
    "return_dict_in_generate",
    "forced_bos_token_id",
    "forced_eos_token_id",
    "remove_invalid_values",
    "architectures",
    "finetuning_task",
    "id2label",
    "label2id",
    "num_labels",
    "task_specific_params",
    "problem_type",
    "bos_token_id",
    "pad_token_id",
    "eos_token_id",
    "decoder_start_token_id",
    "sep_token_id",
    "torchscript",
    "tie_word_embeddings",
    "torch_dtype",
]
HF_TOKENIZER_PARAMS = [
    "pretrained_model_name_or_path",
    "model_max_length",
    "padding_side",
    "truncation_side",
    "chat_template",
    "model_input_names",
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
    "additional_special_tokens",
    "clean_up_tokenization_spaces",
    "split_special_tokens",
    "inputs",
    "config",
    "cache_dir",
    "force_download",
    "resume_download",
    "proxies",
    "revision",
    "subfolder",
    "use_fast",
    "tokenizer_type",
    "trust_remote_code",
    "src_lang",
    "tgt_lang",
]


class ModelFromCheckpoint(PreTrainedModel):
    """Model from Hugging Face checkpoint.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer for the model.
        config (AutoConfig): Configuration for the model.
        model (AutoModelForSeq2SeqLM): Instantiated model.
        custom_device (torch.device or str): Device for model computations.
        generate_kwargs (dict): Keyword arguments to be passed to the model.generate().
    """

    def __init__(self, checkpoint=None, use_cpu=False, **kwargs):
        """Initialize the adapter model and tokenizer with a Hugging Face checkpoint.

        Args:
            checkpoint:
                Specifies the string checkpoint of the model to load. Either a HF
                hub checkpoint or the path to the local model checkpoint directory.
            use_cpu:
                When true, tells the model to make the device 'cpu' even when a GPU is
                available. When false, the model well always attempt to use any available
                GPU(s) available in the runtime.
            **kwargs:
                Keyword arguments to be passed as necessary to the model, model config,
                and tokenizer initialization functions.

        """
        validate_required_params(dict(checkpoint=checkpoint))

        model_kwargs = dict_subset(kwargs, HF_MODEL_FROM_PRETRAINED_PARAMS)
        model_config_kwargs = dict_subset(kwargs, HF_MODEL_CONFIG_PARAMS)
        tokenizer_kwargs = dict_subset(kwargs, HF_TOKENIZER_PARAMS)
        self.generate_kwargs = dict_subset(kwargs, HF_GENERATE_FUNCTION_PARAMS)

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, **tokenizer_kwargs)

        forced_bos_token_tgt_lang = kwargs.get("forced_bos_token_tgt_lang", None)
        if forced_bos_token_tgt_lang:
            model_config_kwargs["forced_bos_token_id"] = self.tokenizer.lang_code_to_id[
                forced_bos_token_tgt_lang
            ]

        if not checkpoint:
            raise ValueError("Missing required parameter 'checkpoint'")
        self.config = AutoConfig.from_pretrained(
            checkpoint,
            **model_config_kwargs,
        )
        super(ModelFromCheckpoint, self).__init__(self.config)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint, config=self.config, **model_kwargs
        )

        self.custom_device = (
            torch.device("cuda")
            if torch.cuda.is_available() and (not use_cpu)
            else "cpu"
        )
        self.model.to(self.custom_device)

        print(
            f"ModelFromCheckpoint: Loaded checkpoint '{checkpoint}'. Using device '{self.custom_device}'."
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass through the model.

        Args:
            input_ids (torch.Tensor): Input tensor.
            attention_mask (torch.Tensor, optional): Attention mask tensor.
            labels (torch.Tensor, optional): Labels tensor.

        Raises:
            NotImplementedError: This method is not implemented in the base class.
        """
        raise NotImplementedError(
            "ModelFromCheckpoint.forward not implemented in base class."
        )
