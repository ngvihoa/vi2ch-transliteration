"""Configuration for the Viet-Han BERT encoder-decoder model."""

from transformers import BertConfig


class VietHanBertConfig(BertConfig):
    """Configuration shared by the BERT encoder and Transformer decoder."""

    model_type = "viet_han_bert"
    
    # Thêm auto_map để HF biết dùng file nào khi load
    auto_map = {
        "AutoConfig": "configuration_viet_han_bert.VietHanBertConfig",
        "AutoModel": "modeling_viet_han_bert.VietHanBertModel",
    }

    def __init__(
        self,
        han_vocab_size: int = None,
        decoder_layers: int = 4,
        decoder_heads: int = 8,
        decoder_ffn_dim: int = 2048,
        max_target_position_embeddings: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.han_vocab_size = han_vocab_size
        self.decoder_layers = decoder_layers
        self.decoder_heads = decoder_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.max_target_position_embeddings = max_target_position_embeddings
