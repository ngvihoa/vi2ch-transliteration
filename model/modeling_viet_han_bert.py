"""PyTorch implementation of the Viet-Han BERT model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from .configuration_viet_han_bert import VietHanBertConfig


class VietHanBertModel(nn.Module):
    """BERT encoder with an autoregressive Transformer decoder."""

    def __init__(self, config: VietHanBertConfig) -> None:
        super().__init__()

        self.config = config
        self.bert = BertModel(config, add_pooling_layer=False)

        self.target_embedding = nn.Embedding(
            config.han_vocab_size,
            config.hidden_size,
        )
        self.target_position_embedding = nn.Embedding(
            config.max_target_position_embeddings,
            config.hidden_size,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_ffn_dim,
            dropout=config.hidden_dropout_prob,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.decoder_layers,
        )
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.han_vocab_size,
            bias=False,
        )

        self.han_pad_id = 0
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        encoder_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        memory = encoder_outputs.last_hidden_state

        if decoder_input_ids is None:
            raise ValueError("decoder_input_ids is required.")

        batch_size, target_len = decoder_input_ids.shape
        positions = torch.arange(
            target_len,
            device=decoder_input_ids.device,
        )
        positions = positions.unsqueeze(0).expand(batch_size, target_len)

        decoder_embeddings = (
            self.target_embedding(decoder_input_ids)
            + self.target_position_embedding(positions)
        )
        causal_mask = torch.triu(
            torch.ones(
                target_len,
                target_len,
                device=decoder_input_ids.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        target_key_padding_mask = decoder_input_ids == self.han_pad_id
        memory_key_padding_mask = (
            attention_mask == 0 if attention_mask is not None else None
        )

        decoder_outputs = self.decoder(
            tgt=decoder_embeddings,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        decoder_outputs = self.final_layer_norm(decoder_outputs)
        logits = self.lm_head(decoder_outputs)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )

        return {
            "loss": loss,
            "logits": logits,
            "encoder_hidden_states": memory,
        }
