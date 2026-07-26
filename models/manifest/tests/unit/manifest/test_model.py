import pytest
import torch

from manifest.model import (
    DataEncoder,
    DecoderBlock,
    EncoderBlock,
    FeedForward,
    LayerNorm,
    MultiHeadAttention,
    RMSNorm,
    TokenDecoder,
    TransformerModel,
)

# Small, fast config for the whole suite. d_model is divisible by n_heads.
D_MODEL = 64
N_HEADS = 4
D_FF = 128
INPUT_DIM = 3
VOCAB = 16
MAX_SEQ_LEN = 32


def _mha(d_model=D_MODEL, n_heads=N_HEADS, dropout=0.0, use_bias=True):
    return MultiHeadAttention(d_model, n_heads, dropout=dropout, use_bias=use_bias)


class TestMultiHeadAttention:
    def test_constructor_rejects_non_divisible_heads(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(D_MODEL, n_heads=5)

    def test_self_attention_output_shape(self):
        mha = _mha()
        x = torch.randn(4, 8, D_MODEL)
        assert mha(x, x, x).shape == (4, 8, D_MODEL)

    def test_cross_attention_output_shape(self):
        mha = _mha()
        q = torch.randn(4, 8, D_MODEL)
        k = torch.randn(4, 6, D_MODEL)
        assert mha(q, k, k).shape == (4, 8, D_MODEL)

    def test_self_path_and_cross_path_agree_for_identical_inputs(self):
        # Self-attn packs QKV in one matmul; cross-attn chunks the packed weight
        # and projects q/k/v separately. With identical inputs and no dropout
        # the two code paths must agree numerically.
        torch.manual_seed(0)
        mha = _mha(use_bias=True)
        mha.eval()
        x = torch.randn(2, 8, D_MODEL)
        out_self = mha(x, x, x)
        out_cross = mha(x, x.clone(), x.clone())
        assert torch.allclose(out_self, out_cross, atol=1e-5)

    def test_key_padding_mask_equivalent_to_removing_masked_keys(self):
        # A masked key contributes zero to softmax numerator and denominator,
        # so masking the last key must reproduce attention over the trimmed
        # (L_k-1) key set. Projection-independent: no parameter surgery needed.
        torch.manual_seed(0)
        mha = _mha()
        mha.eval()
        B, Lq, Lk = 2, 3, 5
        q = torch.randn(B, Lq, D_MODEL)
        k = torch.randn(B, Lk, D_MODEL)
        v = torch.randn(B, Lk, D_MODEL)
        mask = torch.ones(B, Lk, dtype=torch.bool)
        mask[:, -1] = False  # hide the last key
        out_masked = mha(q, k, v, key_padding_mask=mask)
        out_trimmed = mha(q, k[:, :-1], v[:, :-1])
        assert torch.allclose(out_masked, out_trimmed, atol=1e-6)
        # the mask must actually do something, pinning the polarity True=attend
        out_full = mha(q, k, v)
        assert not torch.allclose(out_masked, out_full, atol=1e-6)

    def test_causal_mask_blocks_future_positions(self):
        torch.manual_seed(0)
        mha = _mha()
        mha.eval()
        B, L = 2, 5
        x = torch.randn(B, L, D_MODEL)
        out_ref = mha(x, x, x, is_causal=True)
        x2 = x.clone()
        x2[:, -1, :] = torch.randn(B, D_MODEL) * 100.0
        out_pert = mha(x2, x2, x2, is_causal=True)
        # positions before the last cannot see the perturbed key/value
        assert torch.allclose(out_ref[:, :-1, :], out_pert[:, :-1, :], atol=1e-6)
        # the last position sees everything, so it must change
        assert not torch.allclose(out_ref[:, -1, :], out_pert[:, -1, :], atol=1e-4)

    def test_causal_and_padding_mask_combine(self):
        # Exercises the branch where SDPA cannot take attn_mask + is_causal at
        # once: the module ANDs the padding mask with a causal triangle. Hiding
        # the last key changes only outputs at positions >= that key (causality
        # already excludes it from earlier positions); every earlier position
        # must match causal-only attention.
        torch.manual_seed(0)
        mha = _mha()
        mha.eval()
        B, L = 2, 5
        q = torch.randn(B, L, D_MODEL)
        k = torch.randn(B, L, D_MODEL)
        v = torch.randn(B, L, D_MODEL)
        mask = torch.ones(B, L, dtype=torch.bool)
        mask[:, -1] = False
        out_combined = mha(q, k, v, key_padding_mask=mask, is_causal=True)
        out_causal = mha(q, k, v, is_causal=True)
        # positions 0..L-2: causal already blocks the last key -> identical
        assert torch.allclose(out_combined[:, :-1, :], out_causal[:, :-1, :], atol=1e-6)
        # the last position loses the last key under the padding mask -> differs
        assert not torch.allclose(out_combined[:, -1, :], out_causal[:, -1, :], atol=1e-4)
        assert torch.isfinite(out_combined).all()

    def test_cross_attention_bias_gradients_are_nonzero(self):
        # The cross-attn path manually chunks packed_qkv_proj.bias into
        # q/k/v slices and applies them via F.linear. Verify gradient still
        # reaches the packed bias (and the out_proj bias).
        torch.manual_seed(0)
        mha = _mha(use_bias=True)
        q = torch.randn(2, 4, D_MODEL)
        k = torch.randn(2, 6, D_MODEL)
        v = torch.randn(2, 6, D_MODEL)
        mha(q, k, v).sum().backward()  # distinct q/k/v -> cross-attn path
        packed = mha.packed_qkv_proj
        assert packed.bias is not None
        assert packed.bias.grad is not None
        assert packed.bias.grad.abs().sum() > 0
        assert mha.out_proj.bias.grad is not None
        assert mha.out_proj.bias.grad.abs().sum() > 0

    def test_use_bias_false_omits_bias_parameters(self):
        mha = _mha(use_bias=False)
        assert mha.packed_qkv_proj.bias is None
        assert mha.out_proj.bias is None
        x = torch.randn(2, 4, D_MODEL)
        assert mha(x, x, x).shape == (2, 4, D_MODEL)


class TestLayerNorm:
    def test_output_shape(self):
        out = LayerNorm(D_MODEL)(torch.randn(3, 7, D_MODEL))
        assert out.shape == (3, 7, D_MODEL)

    def test_normalizes_to_zero_mean_unit_variance(self):
        ln = LayerNorm(D_MODEL)
        x = torch.randn(3, 7, D_MODEL) * 5 + 2
        out = ln(x)
        assert out.mean(dim=-1).abs().max() < 1e-5
        std = out.std(dim=-1, unbiased=False)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-2)

    def test_affine_parameters_receive_gradient(self):
        ln = LayerNorm(D_MODEL)
        ln(torch.randn(3, 7, D_MODEL)).sum().backward()
        assert ln.weight.grad.abs().sum() > 0
        assert ln.bias.grad.abs().sum() > 0


class TestRMSNorm:
    def test_output_shape(self):
        out = RMSNorm(D_MODEL)(torch.randn(3, 7, D_MODEL))
        assert out.shape == (3, 7, D_MODEL)

    def test_has_no_bias_parameter(self):
        rn = RMSNorm(D_MODEL)
        assert getattr(rn, "bias", None) is None

    def test_applies_inverse_rms_scaling(self):
        eps = 1e-5
        rn = RMSNorm(D_MODEL, eps=eps)
        x = torch.randn(3, 7, D_MODEL) * 3 + 1
        out = rn(x)
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        assert torch.allclose(out, x / torch.sqrt(ms + eps), atol=1e-6)

    def test_weight_receives_gradient(self):
        rn = RMSNorm(D_MODEL)
        rn(torch.randn(3, 7, D_MODEL)).sum().backward()
        assert rn.weight.grad.abs().sum() > 0


class TestFeedForward:
    def test_output_shape(self):
        out = FeedForward(D_MODEL, D_FF)(torch.randn(3, 7, D_MODEL))
        assert out.shape == (3, 7, D_MODEL)

    def test_gradient_flows_to_both_layers(self):
        ff = FeedForward(D_MODEL, D_FF)
        ff(torch.randn(3, 7, D_MODEL)).sum().backward()
        assert ff.fc1.weight.grad.abs().sum() > 0
        assert ff.fc2.weight.grad.abs().sum() > 0


class TestEncoderBlock:
    def test_output_shape(self):
        blk = EncoderBlock(D_MODEL, N_HEADS, D_FF, dropout=0.0)
        assert blk(torch.randn(2, 8, D_MODEL)).shape == (2, 8, D_MODEL)

    def test_accepts_key_padding_mask(self):
        blk = EncoderBlock(D_MODEL, N_HEADS, D_FF, dropout=0.0)
        mask = torch.ones(2, 8, dtype=torch.bool)
        mask[:, -2:] = False
        out = blk(torch.randn(2, 8, D_MODEL), key_padding_mask=mask)
        assert out.shape == (2, 8, D_MODEL)
        assert torch.isfinite(out).all()

    def test_gradient_flows(self):
        blk = EncoderBlock(D_MODEL, N_HEADS, D_FF, dropout=0.0)
        blk(torch.randn(2, 8, D_MODEL)).sum().backward()
        assert blk.self_attn.packed_qkv_proj.weight.grad.abs().sum() > 0
        assert blk.ffn.fc1.weight.grad.abs().sum() > 0


class TestDecoderBlock:
    def test_output_shape(self):
        blk = DecoderBlock(D_MODEL, N_HEADS, D_FF, dropout=0.0)
        x = torch.randn(2, 8, D_MODEL)
        enc = torch.randn(2, 5, D_MODEL)
        assert blk(x, enc).shape == (2, 8, D_MODEL)

    def test_accepts_data_and_token_masks(self):
        blk = DecoderBlock(D_MODEL, N_HEADS, D_FF, dropout=0.0)
        x = torch.randn(2, 8, D_MODEL)
        enc = torch.randn(2, 5, D_MODEL)
        data_mask = torch.ones(2, 5, dtype=torch.bool)
        token_mask = torch.ones(2, 8, dtype=torch.bool)
        token_mask[:, -1] = False
        out = blk(x, enc, data_mask=data_mask, token_mask=token_mask)
        assert torch.isfinite(out).all()

    def test_gradient_flows_through_all_sublayers(self):
        blk = DecoderBlock(D_MODEL, N_HEADS, D_FF, dropout=0.0)
        x = torch.randn(2, 8, D_MODEL)
        enc = torch.randn(2, 5, D_MODEL)
        blk(x, enc).sum().backward()
        assert blk.self_attn.packed_qkv_proj.weight.grad.abs().sum() > 0
        assert blk.cross_attn.packed_qkv_proj.weight.grad.abs().sum() > 0
        assert blk.ffn.fc1.weight.grad.abs().sum() > 0


class TestDataEncoder:
    def test_output_shape(self):
        enc = DataEncoder(INPUT_DIM, D_MODEL, N_HEADS, D_FF, n_layers=2, dropout=0.0)
        assert enc(torch.randn(2, 10, INPUT_DIM)).shape == (2, 10, D_MODEL)

    def test_accepts_data_mask(self):
        enc = DataEncoder(INPUT_DIM, D_MODEL, N_HEADS, D_FF, n_layers=2, dropout=0.0)
        mask = torch.ones(2, 10, dtype=torch.bool)
        mask[:, -3:] = False
        out = enc(torch.randn(2, 10, INPUT_DIM), data_mask=mask)
        assert torch.isfinite(out).all()

    def test_gradient_flows(self):
        enc = DataEncoder(INPUT_DIM, D_MODEL, N_HEADS, D_FF, n_layers=2, dropout=0.0)
        enc(torch.randn(2, 10, INPUT_DIM)).sum().backward()
        assert enc.data_proj.weight.grad.abs().sum() > 0
        assert enc.blocks[0].self_attn.packed_qkv_proj.weight.grad.abs().sum() > 0


class TestTokenDecoder:
    def _dec(self, dropout=0.0):
        return TokenDecoder(
            VOCAB, MAX_SEQ_LEN, D_MODEL, N_HEADS, D_FF, n_layers=2, dropout=dropout
        )

    def test_output_shapes(self):
        dec = self._dec()
        B, L = 2, 8
        tokens = torch.randint(0, VOCAB, (B, L))
        enc = torch.randn(B, 5, D_MODEL)
        logits, num_preds = dec(tokens, torch.ones(B, L), enc)
        assert logits.shape == (B, L, VOCAB)
        assert num_preds.shape == (B, L, 1)

    def test_num_values_scale_the_embedding(self):
        # num_values multiplies the token embedding per position; changing it
        # must change the output logits.
        torch.manual_seed(0)
        dec = self._dec()
        dec.eval()
        B, L = 2, 8
        tokens = torch.randint(0, VOCAB, (B, L))
        enc = torch.randn(B, 5, D_MODEL)
        logits_a, _ = dec(tokens, torch.ones(B, L), enc)
        nv = torch.ones(B, L)
        nv[:, L // 2] = 5.0
        logits_b, _ = dec(tokens, nv, enc)
        assert not torch.allclose(logits_a, logits_b)

    def test_gradient_flows_to_all_heads(self):
        dec = self._dec()
        B, L = 2, 8
        tokens = torch.randint(0, VOCAB, (B, L))
        enc = torch.randn(B, 5, D_MODEL)
        logits, num_preds = dec(tokens, torch.ones(B, L), enc)
        (logits.sum() + num_preds.sum()).backward()
        assert dec.token_emb.weight.grad.abs().sum() > 0
        assert dec.pos_emb.weight.grad.abs().sum() > 0
        assert dec.logit_head.weight.grad.abs().sum() > 0
        assert dec.numeric_head[0].weight.grad.abs().sum() > 0

    def test_sequence_longer_than_max_seq_len_raises(self):
        dec = self._dec()
        L = MAX_SEQ_LEN + 1
        tokens = torch.randint(0, VOCAB, (1, L))
        with pytest.raises(IndexError):
            dec(tokens, torch.ones(1, L), torch.randn(1, 3, D_MODEL))


class TestTransformerModel:
    def _model(self, dropout=0.0):
        return TransformerModel(
            INPUT_DIM,
            VOCAB,
            MAX_SEQ_LEN,
            D_MODEL,
            N_HEADS,
            D_FF,
            n_enc_layers=2,
            n_dec_layers=2,
            dropout=dropout,
        )

    def _inputs(self, B=2, L=8, Ld=10):
        return (
            torch.randn(B, Ld, INPUT_DIM),
            torch.randint(0, VOCAB, (B, L)),
            torch.ones(B, L),
        )

    def test_forward_output_shapes(self):
        m = self._model()
        data, tokens, num_values = self._inputs()
        logits, num_preds = m(data, tokens, num_values)
        assert logits.shape == (2, 8, VOCAB)
        assert num_preds.shape == (2, 8, 1)

    def test_gradient_flows_to_all_components(self):
        m = self._model()
        data, tokens, num_values = self._inputs()
        logits, num_preds = m(data, tokens, num_values)
        (logits.sum() + num_preds.sum()).backward()
        checks = {
            "encoder.data_proj": m.encoder.data_proj.weight,
            "encoder.block": m.encoder.blocks[0].self_attn.packed_qkv_proj.weight,
            "encoder.final_norm": m.encoder.final_norm.weight,
            "decoder.token_emb": m.decoder.token_emb.weight,
            "decoder.pos_emb": m.decoder.pos_emb.weight,
            "decoder.block.self_attn": m.decoder.blocks[0].self_attn.packed_qkv_proj.weight,
            "decoder.block.cross_attn": m.decoder.blocks[0].cross_attn.packed_qkv_proj.weight,
            "decoder.logit_head": m.decoder.logit_head.weight,
            "decoder.numeric_head": m.decoder.numeric_head[0].weight,
        }
        for name, p in checks.items():
            assert p.grad is not None, f"{name} got no gradient"
            assert p.grad.abs().sum() > 0, f"{name} gradient is all zeros"

    def test_eval_mode_is_deterministic(self):
        m = self._model(dropout=0.3)
        m.eval()
        data, tokens, num_values = self._inputs()
        out1 = m(data, tokens, num_values)
        out2 = m(data, tokens, num_values)
        assert torch.allclose(out1[0], out2[0], atol=1e-6)
        assert torch.allclose(out1[1], out2[1], atol=1e-6)

    def test_masks_are_accepted_end_to_end(self):
        m = self._model()
        B, L, Ld = 2, 8, 10
        data = torch.randn(B, Ld, INPUT_DIM)
        tokens = torch.randint(0, VOCAB, (B, L))
        num_values = torch.ones(B, L)
        data_mask = torch.ones(B, Ld, dtype=torch.bool)
        token_mask = torch.ones(B, L, dtype=torch.bool)
        data_mask[:, -2:] = False
        token_mask[:, -1] = False
        logits, num_preds = m(data, tokens, num_values, data_mask, token_mask)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(num_preds).all()
