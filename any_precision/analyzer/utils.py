import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel, AutoTokenizer, PreTrainedTokenizerBase


def get_model_class(config):
    if config.model_type == 'qwen3_5':
        from transformers import Qwen3_5ForConditionalGeneration
        return Qwen3_5ForConditionalGeneration
    return AutoModelForCausalLM


def build_model(config, dtype, trust_remote_code=True):
    model_class = get_model_class(config)
    if config.model_type == 'qwen3_5':
        return model_class._from_config(config, dtype=dtype)
    return model_class.from_config(config, torch_dtype=dtype,
                                   trust_remote_code=trust_remote_code)


def load_model(model_str_or_model, dtype=torch.float16, cpu_only=False):
    """Returns a model from a string or a model object. If a string is passed, it will be loaded from the HuggingFace"""
    if isinstance(model_str_or_model, str):
        config = AutoConfig.from_pretrained(model_str_or_model, trust_remote_code=True)
        model_class = get_model_class(config)
        if config.model_type == 'qwen3_5':
            dtype = torch.bfloat16
        model = model_class.from_pretrained(
            model_str_or_model,
            config=config,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map='cpu' if cpu_only else 'auto',
        )
    else:
        assert isinstance(model_str_or_model, PreTrainedModel), "model must be a string or a PreTrainedModel"
        model = model_str_or_model
    return model


def load_tokenizer(model_str_or_model_or_tokenizer):
    """Returns a tokenizer from the model string or model object or tokenizer object"""
    if isinstance(model_str_or_model_or_tokenizer, str):
        model_str = model_str_or_model_or_tokenizer
        return AutoTokenizer.from_pretrained(model_str, trust_remote_code=True)
    elif isinstance(model_str_or_model_or_tokenizer, PreTrainedModel):
        model_str = model_str_or_model_or_tokenizer.name_or_path
        return AutoTokenizer.from_pretrained(model_str, trust_remote_code=True)
    else:
        assert isinstance(model_str_or_model_or_tokenizer, PreTrainedTokenizerBase), \
            f"Unsupported type for model_str_or_model_or_tokenizer: {type(model_str_or_model_or_tokenizer)}"
        return model_str_or_model_or_tokenizer
