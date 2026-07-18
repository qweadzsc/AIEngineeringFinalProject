from transformers.configuration_utils import PretrainedConfig


class EConfig(PretrainedConfig):
    model_type = "llama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_scaling=None,
        **kwargs,
    ):
        transformer_layer_config = kwargs.pop("transformer_layer_config", None)
        if transformer_layer_config is not None:
            if isinstance(transformer_layer_config, PretrainedConfig):
                transformer_layer_config = transformer_layer_config.to_dict()
            elif hasattr(transformer_layer_config, "to_dict"):
                transformer_layer_config = transformer_layer_config.to_dict()
            if not isinstance(transformer_layer_config, dict):
                raise ValueError(
                    f"`transformer_layer_config` must be a dict or PretrainedConfig, got {type(transformer_layer_config)}"
                )

            vocab_size = transformer_layer_config.get("vocab_size", vocab_size)
            hidden_size = transformer_layer_config.get("hidden_size", hidden_size)
            intermediate_size = transformer_layer_config.get("intermediate_size", intermediate_size)
            num_hidden_layers = transformer_layer_config.get("num_hidden_layers", num_hidden_layers)
            num_attention_heads = transformer_layer_config.get("num_attention_heads", num_attention_heads)
            num_key_value_heads = transformer_layer_config.get("num_key_value_heads", num_key_value_heads)
            hidden_act = transformer_layer_config.get("hidden_act", hidden_act)
            max_position_embeddings = transformer_layer_config.get(
                "max_position_embeddings", max_position_embeddings
            )
            initializer_range = transformer_layer_config.get("initializer_range", initializer_range)
            rms_norm_eps = transformer_layer_config.get("rms_norm_eps", rms_norm_eps)
            use_cache = transformer_layer_config.get("use_cache", use_cache)
            pad_token_id = transformer_layer_config.get("pad_token_id", pad_token_id)
            bos_token_id = transformer_layer_config.get("bos_token_id", bos_token_id)
            eos_token_id = transformer_layer_config.get("eos_token_id", eos_token_id)
            pretraining_tp = transformer_layer_config.get("pretraining_tp", pretraining_tp)
            tie_word_embeddings = transformer_layer_config.get("tie_word_embeddings", tie_word_embeddings)
            rope_scaling = transformer_layer_config.get("rope_scaling", rope_scaling)

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_scaling = rope_scaling
        self.head_dim = kwargs.pop("head_dim", None)
        if transformer_layer_config is not None and self.head_dim is None:
            self.head_dim = transformer_layer_config.get("head_dim")
        self.rope_theta = kwargs.pop("rope_theta", None)
        if transformer_layer_config is not None and self.rope_theta is None:
            self.rope_theta = transformer_layer_config.get("rope_theta")
        self.draft_vocab_size = kwargs.pop("draft_vocab_size", vocab_size)
        self.target_hidden_size = kwargs.pop("target_hidden_size", None)
        self.norm_before_fc = kwargs.pop("norm_before_fc", False)
        self.norm_before_residual = kwargs.pop("norm_before_residual", False)
        self.eagle_aux_hidden_state_layer_ids = kwargs.pop("eagle_aux_hidden_state_layer_ids", None)
        self._rope_scaling_validation()

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict):
            raise ValueError(f"`rope_scaling` must be a dictionary, got {self.rope_scaling}")

        rope_scaling_type = self.rope_scaling.get("type", self.rope_scaling.get("rope_type", None))
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None:
            raise ValueError(f"`rope_scaling` must contain `type` or `rope_type`, got {self.rope_scaling}")
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, (float, int)) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a number > 1, got {rope_scaling_factor}")
