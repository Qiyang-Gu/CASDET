# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from CASDET.model_adapter import ModelAdapter  # noqa: E402
from CASDET.casdet import run_casdet  # noqa: E402

@pytest.fixture(scope="module")
def tiny_adapter(tmp_path_factory):
    from tokenizers import Tokenizer, models as tok_models
    from transformers import (LlamaConfig, LlamaForCausalLM,
                              PreTrainedTokenizerFast)

    torch.manual_seed(0)
    base = tmp_path_factory.mktemp("tiny_llama")
    config = LlamaConfig(
        vocab_size=1000, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=256, hidden_act="silu", rms_norm_eps=1e-5,
    )
    model = LlamaForCausalLM(config)
    model.save_pretrained(base / "model")

    vocab = {f"tok{i}": i for i in range(100)}
    vocab.update({"<unk>": 100, "<s>": 101, "</s>": 102})
    tok = Tokenizer(tok_models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>",
                                        bos_token="<s>", eos_token="</s>",
                                        pad_token="<unk>")
    tokenizer.save_pretrained(base / "model")

    adapter = ModelAdapter(str(base / "model"), device="cpu", dtype="float32")
    return adapter

def test_capture_shapes(tiny_adapter):
    adapter = tiny_adapter
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    resp_span = (7, 10)
    ctx_mask = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    with adapter.capture(ids, resp_span, ctx_mask) as layers:
        assert set(layers) == {0, 1}
        for li, st in layers.items():
            R, d = 3, adapter.hidden_size
            assert st.attn_out.shape == (R, d)
            assert st.e.shape == (R, d)
            assert st.f.shape == (R, d)
            assert st.x_mid.shape == (R, d)
            assert st.x.shape == (R, d)

def test_context_contribution_equals_full_attention(tiny_adapter):
    adapter = tiny_adapter
    ids = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    resp_span = (5, 10)
    ctx_mask = [1] * len(ids)
    with adapter.capture(ids, resp_span, ctx_mask) as layers:
        for li, st in layers.items():
            diff = (st.e.float() - st.attn_out.float()).abs().max().item()
            assert diff < 1e-3, f"layer {li}: e does not match attn_out (diff={diff})"

def test_ffn_sensitivity_to_modified_input(tiny_adapter):
    adapter = tiny_adapter
    ids = [5, 6, 7, 8, 9, 10, 11, 12]
    resp_span = (5, 8)
    ctx_mask = [1, 1, 1, 1, 1, 0, 0, 0]
    with adapter.capture(ids, resp_span, ctx_mask) as layers:
        st = layers[0]
        f_normal = adapter.run_ffn(0, st.x_mid)
        f_control = adapter.run_ffn(0, st.x_mid - st.e)
        diff = (f_normal - f_control).abs().max().item()
        assert diff > 0.0

def test_x_recursion_consistency(tiny_adapter):
    adapter = tiny_adapter
    ids = [1, 2, 3, 4, 5, 6, 7, 8]
    resp_span = (5, 8)
    ctx_mask = [1, 1, 1, 1, 1, 0, 0, 0]
    with adapter.capture(ids, resp_span, ctx_mask) as layers:
        pred_rows = slice(resp_span[0], resp_span[1])
        x0 = adapter._llm.embed_tokens(
            torch.tensor(ids))[pred_rows]
        expect_mid0 = x0 + layers[0].attn_out
        assert (layers[0].x_mid - expect_mid0).abs().max().item() < 1e-5
        assert (layers[0].x - (layers[0].x_mid + layers[0].f)).abs().max() < 1e-5
        expect_mid1 = layers[0].x_mid + layers[0].f + layers[1].attn_out
        assert (layers[1].x_mid - expect_mid1).abs().max().item() < 1e-5

def test_capture_uses_response_token_rows(tiny_adapter):
    ids = [11, 12, 13, 14, 15, 16]
    resp_span = (4, 6)
    ctx_mask = [1, 1, 1, 1, 0, 0]
    with tiny_adapter.capture(ids, resp_span, ctx_mask) as layers:
        assert all(st.attn_out.shape[0] == resp_span[1] - resp_span[0]
                   for st in layers.values())
        st = layers[tiny_adapter.num_layers - 1]
        lens = tiny_adapter.logit_lens(st.x)
        full = tiny_adapter.forward_logits(torch.tensor(ids))
        pred = full[resp_span[0]:resp_span[1]]
        assert torch.allclose(lens.float(), pred.float(), atol=2e-3, rtol=2e-3)

def test_run_casdet_reads_at_the_answer_position_not_the_prediction_position(
        tiny_adapter, monkeypatch):
    seen = {}
    real_capture = tiny_adapter.capture

    def spy(input_ids, resp_span, ctx_mask, **kwargs):
        seen["span"] = resp_span
        return real_capture(input_ids, resp_span, ctx_mask, **kwargs)

    monkeypatch.setattr(tiny_adapter, "capture", spy)
    ids = [11, 22, 33, 44, 55, 66, 77, 88]
    run_casdet(tiny_adapter, ids, (5, 8), [1, 1, 1, 1, 1, 0, 0, 0],
              sample_id="s", label=False)
    assert seen["span"] == (5, 8), (
        "capture changed the response span; CASDET reads the response positions themselves, "
        "not the shifted prediction positions.")

def test_capture_rows_predict_the_next_token(tiny_adapter):
    adapter = tiny_adapter
    ids = [11, 22, 33, 44, 55, 66, 77, 88]
    resp_span = (5, 8)
    full = adapter.forward_logits(torch.tensor(ids))
    with adapter.capture(ids, resp_span, [1, 1, 1, 1, 1, 0, 0, 0]) as layers:
        lens = adapter.logit_lens(layers[adapter.num_layers - 1].x)
    assert torch.allclose(lens.float(), full[resp_span[0]:resp_span[1]].float(),
                          atol=2e-3, rtol=2e-3)

def test_logit_lens_shape_and_context_contribution_api(tiny_adapter):
    adapter = tiny_adapter
    x = torch.randn(3, adapter.hidden_size)
    logits = adapter.logit_lens(x)
    assert logits.shape == (3, adapter.vocab_size)

    ids = [1, 2, 3, 4, 5, 6, 7, 8]
    resp_span = (5, 8)
    ctx_mask = [0] * len(ids)
    with adapter.capture(ids, resp_span, ctx_mask) as layers:
        assert layers[0].e.abs().max().item() == 0.0
