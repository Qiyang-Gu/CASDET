# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from CASDET.datasets import (  # noqa: E402
    RagSample,
    extract_context_span,
    load_hallurag,
    load_ragtruth,
    selection_eval_splits,
    tokenize_and_align,
)

class MockTokenizer:
    chat_template = "<identity>"

    def __init__(self, vocab, chat_wrap=None):
        self.vocab = vocab
        self._wrap = chat_wrap or (lambda t: t)

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **_):
        return self._wrap(messages[-1]["content"])

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False,
                 **_):
        words = text.split(" ")
        ids, offsets = [], []
        pos = 0
        for w in words:
            s = text.find(w, pos)
            ids.append(self.vocab.get(w, 0))
            offsets.append([s, s + len(w)])
            pos = s + len(w) + 1
        if not words:
            ids, offsets = [], []
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

VOCAB = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6,
         "g": 7, "h": 8, "Context:": 10, "Question:": 11, "Answer:": 12}
TOK = MockTokenizer(VOCAB)

def make_sample(prompt="Context: a b c d Question: e f Answer: ",
                answer="g h") -> RagSample:
    return RagSample(sample_id="s1", query="e f", context="a b c d",
                     answer=answer, label=False, prompt=prompt)

def test_extract_context_span_marker():
    prompt = "### Context: hello world\n### Question: what?"
    ctx, query = extract_context_span(prompt)
    assert ctx == "hello world"
    assert "what?" in query

def test_extract_context_span_fallback():
    ctx, query = extract_context_span("some docs here\nQuestion: why?")
    assert ctx == "some docs here"
    assert "why?" in query

def test_token_spans_alignment():
    s = make_sample()
    s = tokenize_and_align(s, TOK)
    assert s.ctx_token_span == (1, 5)
    assert s.resp_token_span == (len(s.input_ids) - 2, len(s.input_ids))
    assert s.query_token_span[0] == 0
    mask = s.context_mask()
    assert sum(mask) == 4
    assert mask[0] == 0 and mask[1] == 1 and mask[4] == 1 and mask[5] == 0

def test_chat_prefix_shifts_spans():
    wrap = lambda t: "<s> [INST] " + t + " [/INST] "          # noqa: E731
    tok = MockTokenizer(VOCAB, chat_wrap=wrap)
    s = tokenize_and_align(make_sample(), tok)
    r0, r1 = s.resp_token_span
    assert r1 == len(s.input_ids)
    assert [s.input_ids[i] for i in range(r0, r1)] == [VOCAB["g"], VOCAB["h"]]
    c0, c1 = s.ctx_token_span
    assert [s.input_ids[i] for i in range(c0, c1)] == \
        [VOCAB[k] for k in ("a", "b", "c", "d")]
    plain = tokenize_and_align(make_sample(), MockTokenizer(VOCAB))
    assert plain.resp_token_span[0] < r0

class _RoleTokenizer:
    def __init__(self, supports_system: bool):
        self.chat_template = "role" if supports_system else "plain"
        self.supports_system = supports_system
        self.calls = 0

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        self.calls += 1
        if not self.supports_system and any(m["role"] == "system"
                                            for m in messages):
            from jinja2.exceptions import TemplateError
            raise TemplateError("Conversation roles must alternate ...")
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>"
                       for m in messages) + "<assistant>"

def test_chat_prefix_replays_generation_roles():
    from CASDET.datasets import chat_prefix
    msgs = [{"role": "system", "content": "S"},
            {"role": "user", "content": "U"}]
    tok = _RoleTokenizer(supports_system=True)
    assert chat_prefix(tok, "S\n\nU", msgs) == \
        "<system>S</system><user>U</user><assistant>"
    assert chat_prefix(tok, "S\n\nU") == "<user>S\n\nU</user><assistant>"

def test_chat_prefix_merges_when_template_lacks_system_role():
    from CASDET.datasets import chat_prefix, _NO_SYSTEM_ROLE
    tok = _RoleTokenizer(supports_system=False)
    msgs = [{"role": "system", "content": "S"},
            {"role": "user", "content": "U"}]
    assert chat_prefix(tok, "S\n\nU", msgs) == "<user>S\n\nU</user><assistant>"
    assert "plain" in _NO_SYSTEM_ROLE

def test_missing_chat_template_raises():
    class BaseTokenizer(MockTokenizer):
        chat_template = None

    with pytest.raises(ValueError, match="chat_template"):
        tokenize_and_align(make_sample(), BaseTokenizer(VOCAB))

class ChunkTokenizer:

    chat_template = "<identity>"

    def __init__(self, size: int = 4, chat_wrap=None):
        self.size = size
        self._wrap = chat_wrap or (lambda t: t)

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **_):
        return self._wrap(messages[-1]["content"])

    def __call__(self, text, add_special_tokens=False,
                 return_offsets_mapping=False, **_):
        ids, offsets = [], []
        for k, i in enumerate(range(0, len(text), self.size)):
            offsets.append([i, min(i + self.size, len(text))])
            ids.append(k + 1)
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

def test_resp_span_uses_char_offsets_not_length_subtraction():
    s = RagSample(sample_id="s", query="", context="", answer="hijkl",
                  label=False, prompt="abcdefg")
    s = tokenize_and_align(s, ChunkTokenizer(4))
    assert s.input_ids == [1, 2, 3]
    assert s.resp_token_span == (2, 3), "the response span should start at the first token fully inside the answer"

def test_resp_span_covers_all_answer_tokens():
    tk = ChunkTokenizer(4)
    for prompt, answer in [("abc", "defghij"), ("abcdefg", "hij"),
                           ("ab", "cdefghijkl")]:
        s = RagSample(sample_id="s", query="", context="", answer=answer,
                      label=False, prompt=prompt)
        s = tokenize_and_align(s, tk)
        enc = tk(prompt + answer, return_offsets_mapping=True)
        r0, r1 = s.resp_token_span
        assert r1 == len(s.input_ids)
        inside = [i for i, (a, _) in enumerate(enc["offset_mapping"])
                  if a >= len(prompt)]
        assert r0 == min(inside), f"{prompt!r}+{answer!r}: expected start {min(inside)}"

def test_token_span_truncation_shifts():
    s = make_sample()
    s = tokenize_and_align(s, TOK, max_len=7)
    assert len(s.input_ids) == 7
    assert s.input_ids == [10, 11, 5, 6, 12, 7, 8]
    assert s.ctx_token_span == (1, 1)
    assert s.resp_token_span == (5, 7)
    assert s.input_ids[s.resp_token_span[0]] == VOCAB["g"]
    assert s.input_ids[s.resp_token_span[0] - 1] == VOCAB["Answer:"]

def test_no_context_sample():
    s = RagSample(sample_id="x", query="q", context="", answer="h",
                  label=True, prompt="Question: q Answer:")
    s = tokenize_and_align(s, TOK)
    assert s.ctx_token_span == (0, 0)
    assert sum(s.context_mask()) == 0

def test_load_ragtruth_labels(tmp_path):
    rec = {"id": "r1", "question": "q", "prompt": "Context: a\nQuestion: q\nAnswer:",
           "response": "a", "hallucinated": "true", "task": "QA"}
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    samples = load_ragtruth(p)
    assert len(samples) == 1
    assert samples[0].label is True
    assert samples[0].context == "a"

def test_load_hallurag_answerable_only(tmp_path):
    records = [
        {"id": 1, "question": "q1", "contexts": ["c1"], "answer": "a1",
         "label": 1, "answerable": True},
        {"id": 2, "question": "q2", "contexts": ["c2"], "answer": "a2",
         "label": 0, "answerable": False},
    ]
    p = tmp_path / "h.json"
    p.write_text(json.dumps(records), encoding="utf-8")
    samples = load_hallurag(p, answerable_only=True)
    assert len(samples) == 1
    assert samples[0].sample_id == "1"
    samples_all = load_hallurag(p, answerable_only=False)
    assert len(samples_all) == 2

def test_load_ragtruth_quality_filter(tmp_path):
    base = {"question": "q", "prompt": "Question: q\nAnswer:", "response": "a",
            "hallucinated": False, "task": "QA"}
    recs = [dict(base, id="k", quality="good"),
            dict(base, id="r", quality="incorrect_refusal"),
            dict(base, id="t", quality="truncated"),
            dict(base, id="n")]
    p = tmp_path / "q.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    kept = load_ragtruth(p)
    assert [s.sample_id for s in kept] == ["k", "n"]
    assert len(load_ragtruth(p, quality=None)) == 4
    assert len(load_ragtruth(p, quality="truncated")) == 1

def test_ragtruth_context_from_source_info(tmp_path):
    rec = {"id": "r2", "question": "q", "prompt": "Question: q\nAnswer:",
           "response": "x", "hallucinated": False,
           "source_info": [{"source": "w", "text": "doc text"}]}
    p = tmp_path / "r2.jsonl"
    p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    samples = load_ragtruth(p)
    assert samples[0].context == "doc text"
    assert samples[0].label is False

def _mk(sid: str, split: str, label: bool) -> RagSample:
    return RagSample(sample_id=sid, query="q", context="c", answer="a",
                     prompt="p", label=label, split=split)

def test_selection_eval_splits_official_split():
    samples = ([_mk(f"tr{i}", "train", i % 2 == 0) for i in range(10)]
               + [_mk(f"va{i}", "val", i % 2 == 0) for i in range(4)]
               + [_mk(f"te{i}", "test", i % 2 == 0) for i in range(6)])
    sp = selection_eval_splits(samples, sel_ratio=0.2, seed=42)
    assert [s.sample_id for s in sp.evaluation] == [f"te{i}" for i in range(6)]
    assert len(sp.selection) == 14
    assert sp.folds is None

def test_selection_eval_splits_test_only_carves_holdout():
    samples = [_mk(f"te{i}", "test", i % 2 == 0) for i in range(50)]
    sp = selection_eval_splits(samples, sel_ratio=0.2, seed=42)
    assert len(sp.selection) == 10
    assert len(sp.evaluation) == 40
    ids = [s.sample_id for s in sp.evaluation] + [s.sample_id for s in sp.selection]
    assert sorted(ids) == sorted(f"te{i}" for i in range(50))
    again = selection_eval_splits(samples, sel_ratio=0.2, seed=42)
    assert ([s.sample_id for s in again.evaluation]
            == [s.sample_id for s in sp.evaluation])

def test_selection_eval_splits_cv_folds_uses_all_samples():
    samples = [_mk(f"s{i}", "train", i % 3 == 0) for i in range(30)]
    sp = selection_eval_splits(samples, cv_folds=5, seed=42)
    assert len(sp.evaluation) == 30
    assert len(sp.selection) == 30
    assert sp.folds is not None and len(sp.folds) == 5
    assert sum(len(f) for f in sp.folds) == 30
    assert sorted(x for f in sp.folds for x in f) == sorted(
        s.sample_id for s in samples)
