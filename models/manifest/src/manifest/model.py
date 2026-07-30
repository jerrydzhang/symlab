import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import PAD_ID, BOS_ID, EOS_ID, NUM_ID


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, dropout: float = 0.1, use_bias: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.head_dim = d_model // n_heads

        self.packed_qkv_proj = nn.Linear(d_model, 3 * d_model, bias=use_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if query is key and key is value:
            result = self.packed_qkv_proj(query)
            query, key, value = torch.chunk(result, 3, dim=-1)
        else:
            q_weight, k_weight, v_weight = torch.chunk(
                self.packed_qkv_proj.weight, 3, dim=0
            )
            bias = self.packed_qkv_proj.bias
            if bias is not None:
                q_bias, k_bias, v_bias = torch.chunk(bias, 3, dim=0)
            else:
                q_bias = k_bias = v_bias = None
            query, key, value = (
                F.linear(query, q_weight, q_bias),
                F.linear(key, k_weight, k_bias),
                F.linear(value, v_weight, v_bias),
            )

        # (B, L, d_model) -> (B, n_heads, L, head_dim)
        query = query.unflatten(-1, (self.n_heads, self.head_dim)).transpose(1, 2)
        key = key.unflatten(-1, (self.n_heads, self.head_dim)).transpose(1, 2)
        value = value.unflatten(-1, (self.n_heads, self.head_dim)).transpose(1, 2)

        # Expand key_padding_mask (B, L_k) → (B, 1, 1, L_k) for SDPA
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
            # SDPA disallows attn_mask + is_causal together.
            if is_causal:
                L = query.size(-2)
                causal = torch.ones(L, L, device=query.device, dtype=torch.bool).tril()
                attn_mask = attn_mask & causal
                is_causal = False

        attn_outputs = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # (B, n_heads, L, head_dim) -> (B, L, d_model)
        attn_outputs = attn_outputs.transpose(1, 2).flatten(-2)
        attn_outputs = self.out_proj(attn_outputs)

        return attn_outputs


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * (
            x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        )


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.norm1(x)
        attn_output = self.self_attn(
            h, h, h, key_padding_mask=key_padding_mask, is_causal=False
        )
        x = x + self.dropout1(attn_output)

        h = self.norm2(x)
        ffn_output = self.ffn(h)
        x = x + self.dropout2(ffn_output)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        data_mask: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        attn_output = self.self_attn(
            h, h, h, key_padding_mask=token_mask, is_causal=True
        )
        x = x + self.dropout1(attn_output)

        h = self.norm2(x)
        cross_attn_output = self.cross_attn(
            h, enc_output, enc_output, key_padding_mask=data_mask, is_causal=False
        )
        x = x + self.dropout2(cross_attn_output)

        h = self.norm3(x)
        ffn_output = self.ffn(h)
        x = x + self.dropout3(ffn_output)

        return x


class DataEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.data_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.final_norm = LayerNorm(d_model)

    def forward(
        self, data: torch.Tensor, data_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.data_proj(data)
        for block in self.blocks:
            x = block(x, key_padding_mask=data_mask)
        x = self.final_norm(x)
        return x


class TokenDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.final_norm = LayerNorm(d_model)
        self.logit_head = nn.Linear(d_model, vocab_size)
        self.numeric_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, 1),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        num_values: torch.Tensor,
        enc_output: torch.Tensor,
        data_mask: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.token_emb(tokens)
        x = x * num_values.unsqueeze(-1)

        seq_len = tokens.size(1)
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        x = x + self.pos_emb(positions)

        for block in self.blocks:
            x = block(x, enc_output, data_mask=data_mask, token_mask=token_mask)

        x = self.final_norm(x)
        token_logits = self.logit_head(x)
        num_preds = self.numeric_head(x)

        return token_logits, num_preds


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_enc_layers: int,
        n_dec_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.encoder = DataEncoder(
            input_dim, d_model, n_heads, d_ff, n_enc_layers, dropout
        )
        self.decoder = TokenDecoder(
            vocab_size, max_seq_len, d_model, n_heads, d_ff, n_dec_layers, dropout
        )

    def forward(
        self,
        data: torch.Tensor,
        tokens: torch.Tensor,
        num_values: torch.Tensor,
        data_mask: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enc_output = self.encoder(data, data_mask=data_mask)
        token_logits, num_preds = self.decoder(
            tokens, num_values, enc_output, data_mask=data_mask, token_mask=token_mask
        )
        return token_logits, num_preds

    @torch.no_grad()
    def generate(
        self,
        data: torch.Tensor,
        data_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = data.device
        batch_size = data.size(0)
        gen_num_values = torch.ones(
            (batch_size, self.max_seq_len), dtype=torch.float, device=device
        )
        gen_tokens = torch.full(
            (batch_size, self.max_seq_len),
            fill_value=PAD_ID,
            dtype=torch.long,
            device=device,
        )
        gen_tokens[:, 0] = BOS_ID

        enc_output = self.encoder(data, data_mask=data_mask)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for t in range(1, self.max_seq_len):
            token_logits, num_preds = self.decoder(
                gen_tokens[:, :t],
                gen_num_values[:, :t],
                enc_output,
                data_mask=data_mask,
            )

            next_token = token_logits[:, t - 1, :].argmax(dim=-1)
            next_token = torch.masked_fill(next_token, finished, PAD_ID)

            is_num = next_token == NUM_ID
            gen_num_values[:, t] = torch.where(
                is_num, num_preds[:, t - 1, 0], torch.ones(batch_size, device=device)
            )

            gen_tokens[:, t] = next_token

            finished |= next_token == EOS_ID
            if finished.all():
                break

        return gen_tokens, gen_num_values
