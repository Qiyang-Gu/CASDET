# -*- coding: utf-8 -*-

"""Dataset loading and token-position alignment for RAGTruth and HalluRAG."""
from __future__ import annotations

import collections
import json
import logging
import random
from dataclasses import dataclass

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from jinja2.exceptions import TemplateError

PathLike = str | Path

logger = logging.getLogger("CASDET")

@dataclass
class RagSample:
    """A RAG sample with token-level alignment information."""
    sample_id: str
    query: str
    context: str
    answer: str
    label: bool
    prompt: str
    prompt_messages: Optional[List[Dict[str, str]]] = None
    task: str = ""
    split: str = ""
    input_ids: Optional[List[int]] = None
    ctx_token_span: Optional[Tuple[int, int]] = None
    query_token_span: Optional[Tuple[int, int]] = None
    resp_token_span: Optional[Tuple[int, int]] = None

    def context_mask(self) -> List[int]:
        """Binary mask marking the external-context token positions."""
        assert self.ctx_token_span is not None, "call tokenize_and_align first"
        mask = [0] * len(self.input_ids)
        start, end = self.ctx_token_span
        for i in range(max(0, start), min(end, len(mask))):
            mask[i] = 1
        return mask

def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        return s in {"true", "yes", "1", "hallucinated", "non-faithful"}
    return bool(value)

_SUBSET_SEED = 42

def _stratified_subset(cand: List[Tuple[int, Dict[str, Any]]],
                       n: int, stratum: Callable[[Dict[str, Any]], Any],
                       seed: int = _SUBSET_SEED
                       ) -> List[Tuple[int, Dict[str, Any]]]:
    if len(cand) <= n:
        return cand
    groups: Dict[Any, List[int]] = {}
    for pos, (_, rec) in enumerate(cand):
        groups.setdefault(stratum(rec), []).append(pos)
    total = len(cand)
    exact = {k: n * len(v) / total for k, v in groups.items()}
    quota = {k: min(len(groups[k]), max(1, int(exact[k]))) for k in groups}
    order = sorted(groups, key=lambda k: exact[k] - quota[k], reverse=True)
    guard = 0
    while sum(quota.values()) > n and guard < 10000:
        k = order[guard % len(order)]
        if quota[k] > 1:
            quota[k] -= 1
        guard += 1
    guard = 0
    while sum(quota.values()) < n and guard < 10000:
        k = order[guard % len(order)]
        if quota[k] < len(groups[k]):
            quota[k] += 1
        guard += 1
    rng = random.Random(seed)
    picked: List[int] = []
    for k, idxs in groups.items():
        picked += rng.sample(idxs, quota[k])
    return [cand[i] for i in sorted(picked)]

def load_ragtruth(
    path: PathLike,
    max_samples: Optional[int] = None,
    question_key: str = "question",
    response_key: str = "response",
    hallucinated_key: str = "hallucinated",
    prompt_key: str = "prompt",
    id_key: str = "id",
    context_key: Optional[str] = "source_info",
    split: Optional[str] = None,
    quality: Optional[str] = "good",
) -> List[RagSample]:
    """Load RAGTruth records for one model."""
    path = Path(path)
    assert path.exists(), f"data file not found: {path}"
    dropped: collections.Counter = collections.Counter()

    def _iter_records():
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    if line.startswith("[") and line_no == 0:
                        continue
                    raise ValueError(f"{path}: invalid JSON on line {line_no + 1}")
                if split is not None and rec.get("split", "test") != split:
                    continue
                if rec.get(response_key) is None:
                    continue
                q = str(rec.get("quality") or "good")
                if quality is not None and q != quality:
                    dropped[q] += 1
                    continue
                yield line_no, rec

    def _build(line_no: int, rec: Dict[str, Any]) -> RagSample:
        q = rec.get(question_key, "")
        ans = str(rec[response_key])
        prompt = str(rec.get(prompt_key) or _default_prompt(q, ans))
        ctx = _extract_context_field(rec, context_key)
        if not ctx:
            ctx, _ = extract_context_span(prompt)
        sample_id = str(rec.get(id_key) or rec.get("question_id") or
                        rec.get("query_id") or line_no)
        return RagSample(
            sample_id=sample_id,
            query=str(q),
            context=ctx,
            answer=ans,
            label=_to_bool(rec.get(hallucinated_key, False)),
            prompt=prompt,
            task=str(rec.get("task") or rec.get("task_type") or ""),
            split=str(rec.get("split") or ""),
        )

    if max_samples is None:
        out = [_build(line_no, rec) for line_no, rec in _iter_records()]
    else:
        pool = _stratified_subset(
            list(_iter_records()), max_samples,
            lambda rec: (str(rec.get("task") or ""),
                         _to_bool(rec.get(hallucinated_key, False))))
        out = [_build(line_no, rec) for line_no, rec in pool]
    if dropped:
        detail = ", ".join(f"{k}: {v}" for k, v in sorted(dropped.items()))
        logger.info("quality != %r: dropped %d records (%s), kept %d -- %s",
                    quality, sum(dropped.values()), detail, len(out), path.name)
    return out

def _default_prompt(query: str, answer: str) -> str:
    return f"Question: {query}\nAnswer: {answer}"

def _extract_context_field(rec: Dict[str, Any], key: Optional[str]) -> str:
    if not key:
        return ""
    value = rec.get(key)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for k in ("text", "content", "passage", "snippet"):
                    if item.get(k):
                        parts.append(str(item[k]))
                        break
        return "\n".join(parts)
    if isinstance(value, dict):
        for k in ("text", "content", "context", "passage"):
            if value.get(k):
                return str(value[k])
    return ""

def _to_messages(value: Any) -> Optional[List[Dict[str, str]]]:
    if not isinstance(value, list) or not value:
        return None
    messages: List[Dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        role, content = item.get("role"), item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            return None
        messages.append({"role": role, "content": content})
    return messages

def load_hallurag(
    path: PathLike,
    max_samples: Optional[int] = None,
    answerable_only: bool = True,
    question_key: str = "question",
    contexts_key: str = "contexts",
    answer_key: str = "answer",
    label_key: str = "label",
    answerable_key: str = "answerable",
    id_key: str = "id",
    split: Optional[str] = None,
    quality: Optional[str] = "good",
) -> List[RagSample]:
    """Load HalluRAG records for one model."""
    samples: List[RagSample] = []
    path = Path(path)
    assert path.exists(), f"data file not found: {path}"
    records = _load_json_records(path)

    def _eligible(rec: Dict[str, Any]) -> bool:
        if split is not None and str(rec.get("split", "test")) != split:
            return False
        if quality is not None and str(rec.get("quality") or "good") != quality:
            return False
        if answerable_only and not _to_bool(rec.get(answerable_key, True)):
            return False
        return rec.get(answer_key) is not None

    if max_samples is not None:
        cand = _stratified_subset(
            [(i, r) for i, r in enumerate(records) if _eligible(r)], max_samples,
            lambda rec: _to_bool(rec.get(label_key,
                                         rec.get("hallucinated", False))))
    else:
        cand = list(enumerate(records))

    for idx, rec in cand:
        if not _eligible(rec):
            continue
        ans = rec.get(answer_key)
        if ans is None:
            continue
        ctx_list = rec.get(contexts_key) or []
        if isinstance(ctx_list, str):
            ctx_list = [ctx_list]
        ctx = "\n\n".join(str(c) for c in ctx_list)
        q = str(rec.get(question_key, ""))
        ans = str(ans)
        sample_id = str(rec.get(id_key) or rec.get("question_id") or
                        rec.get("qid") or idx)
        prompt = rec.get("prompt") or _default_prompt(q, ans)
        samples.append(RagSample(
            sample_id=sample_id,
            query=q,
            context=ctx,
            answer=ans,
            label=_to_bool(rec.get(label_key, rec.get("hallucinated", False))),
            prompt=str(prompt),
            prompt_messages=_to_messages(rec.get("prompt_messages")),
            task=str(rec.get("task", "")),
            split=str(rec.get("split") or ""),
        ))
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples

def _load_json_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    records: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if isinstance(rec, dict):
                records.append(rec)
        except json.JSONDecodeError:
            continue
    return records

def load_dataset(
    name: str,
    path: PathLike,
    max_samples: Optional[int] = None,
    **kwargs: Any,
) -> List[RagSample]:
    """Load a dataset by name."""
    name = name.lower()
    if name in ("ragtruth", "rag_truth"):
        return load_ragtruth(path, max_samples=max_samples, **kwargs)
    if name in ("hallurag", "hallu_rag"):
        return load_hallurag(path, max_samples=max_samples, **kwargs)
    raise ValueError(f"unknown dataset: {name} (expected ragtruth or hallurag)")

CONTEXT_MARKER_CANDIDATES: Sequence[Tuple[str, str]] = [
    ("### Context:", "### Question:"),
    ("Context:", "Question:"),
    ("### Context", "### Question"),
    ("context:", "question:"),
    ("Passage:", "Question:"),
    ("Retrieved Documents:", "Question:"),
    ("Documents:", "Question:"),
    ("passage 1:", "In case the passages do not contain"),
    ("Summarize the following news within", "output:"),
    ("Structured data:", "Overview:"),
]

def extract_context_span(
    prompt: str,
    candidates: Sequence[Tuple[str, str]] = CONTEXT_MARKER_CANDIDATES,
) -> Tuple[str, str]:
    """Extract the context substring from a prompt."""
    for start_marker, end_marker in candidates:
        s = prompt.find(start_marker)
        if s < 0:
            continue
        body_start = s + len(start_marker)
        e = prompt.find(end_marker, body_start)
        body_end = len(prompt) if e < 0 else e
        ctx = prompt[body_start:body_end].strip()
        if ctx:
            query = prompt[body_end:].strip() if e >= 0 else ""
            return ctx, query
    q_idx = prompt.lower().find("question")
    if q_idx > 0:
        return prompt[:q_idx].strip(), prompt[q_idx:].strip()
    if q_idx == 0:
        return "", prompt.strip()
    return prompt.strip(), ""

CHAT_PROMPT_CHAR_CAP = 12000

_NO_SYSTEM_ROLE: set = set()

def chat_prefix(tokenizer: Any, prompt: str,
                messages: Optional[Sequence[Dict[str, str]]] = None) -> str:
    """Build the chat-template prefix for a prompt."""
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "Tokenizer has no chat_template. CASDET builds inputs with the chat format "
            "of the instruct checkpoints; base models are not supported.")
    if messages and len(prompt) <= CHAT_PROMPT_CHAR_CAP:
        try:
            return tokenizer.apply_chat_template(
                [dict(m) for m in messages],
                tokenize=False, add_generation_prompt=True)
        except (TemplateError, ValueError, TypeError, KeyError) as e:
            tmpl = str(getattr(tokenizer, "chat_template", ""))
            if tmpl not in _NO_SYSTEM_ROLE:
                _NO_SYSTEM_ROLE.add(tmpl)
                logger.info("Falling back to a single user message (role %s is not supported by "
                            "this template: %s: %s)",
                            [m.get("role") for m in messages],
                            type(e).__name__, e)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt[:CHAT_PROMPT_CHAR_CAP]}],
        tokenize=False, add_generation_prompt=True)

def tokenize_and_align(
    sample: RagSample,
    tokenizer: Any,
    max_len: int = 4096,
    add_special_tokens: bool = False,
) -> RagSample:
    """Tokenize a sample and align its query, context, and response token spans."""
    prefix = chat_prefix(tokenizer, sample.prompt, sample.prompt_messages)
    if len(sample.prompt) > CHAT_PROMPT_CHAR_CAP:
        logger.warning("sample %s: prompt length %d exceeds the %d-character limit; "
                       "the excess is not fed to the model. If the context span falls in "
                       "the truncated part, alignment falls back to marker matching.",
                       sample.sample_id, len(sample.prompt),
                       CHAT_PROMPT_CHAR_CAP)
    text = prefix + sample.answer
    ctx_text = sample.context if sample.context else ""
    ctx_char_start = text.find(ctx_text) if ctx_text else -1
    if ctx_char_start < 0:
        ctx_text, _ = extract_context_span(sample.prompt)
        ctx_char_start = text.find(ctx_text) if ctx_text else -1
    ctx_char_end = (ctx_char_start + len(ctx_text)) if ctx_char_start >= 0 else -1

    enc = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        return_offsets_mapping=True,
    )
    ids: List[int] = list(enc["input_ids"])
    offsets = enc.get("offset_mapping")
    if offsets is None:
        raise ValueError(
            "Tokenizer did not return offset_mapping. Character-offset alignment is required: "
            "subtracting lengths assumes additive tokenization, but BPE can merge tokens "
            "across the prompt/response boundary, so no fallback is provided.")

    ans_char_start = len(prefix)
    # Map character spans to token spans.
    resp_tokens = [i for i, (a, b) in enumerate(offsets)
                   if a >= ans_char_start and b > a]
    if not resp_tokens and sample.answer.strip():
        resp_tokens = [i for i, (a, b) in enumerate(offsets)
                       if b > ans_char_start]
        if resp_tokens:
            logger.warning("sample %s: the first response token was merged with the last prefix token; "
                           "using the cross-boundary token for the response span", sample.sample_id)
    resp_start = min(resp_tokens) if resp_tokens else len(ids)
    resp_span = (resp_start, len(ids))

    if ctx_char_start >= 0 and ctx_char_end > ctx_char_start:
        ctx_tokens = [i for i, (a, b) in enumerate(offsets)
                      if b > ctx_char_start and a < ctx_char_end]
        ctx_span = (min(ctx_tokens), max(ctx_tokens) + 1) if ctx_tokens else (0, 0)
    else:
        ctx_span = (0, 0)

    q_tokens = [i for i in range(resp_start)
                if not (ctx_span[0] <= i < ctx_span[1])]
    query_span = (min(q_tokens), max(q_tokens) + 1) if q_tokens else (0, 0)

    if len(ids) > max_len:
        overflow = len(ids) - max_len
        cs, ce = ctx_span
        cut = min(overflow, max(0, ce - cs))
        del ids[cs:cs + cut]
        sample.input_ids = ids
        sample.ctx_token_span = (cs, max(cs, ce - cut))
        sample.query_token_span = (max(0, query_span[0] - cut),
                                   max(0, query_span[1] - cut))
        sample.resp_token_span = (max(0, resp_span[0] - cut),
                                  max(0, resp_span[1] - cut))
    else:
        sample.input_ids = ids
        sample.ctx_token_span = ctx_span
        sample.query_token_span = query_span
        sample.resp_token_span = resp_span
    return sample

@dataclass
class EvaluationSplit:
    """Selection/evaluation split, optionally with cross-validation folds."""
    evaluation: List[RagSample]
    selection: List[RagSample]
    folds: Optional[List[List[str]]] = None

def _stratified_folds(samples: Sequence[RagSample], k: int,
                      seed: int) -> List[List[RagSample]]:
    rng = random.Random(seed)
    folds: List[List[RagSample]] = [[] for _ in range(k)]
    for group in ([s for s in samples if s.label],
                  [s for s in samples if not s.label]):
        shuffled = list(group)
        rng.shuffle(shuffled)
        for i, s in enumerate(shuffled):
            folds[i % k].append(s)
    return folds

def selection_eval_splits(
    samples: Sequence[RagSample],
    sel_ratio: float = 0.2,
    seed: int = 42,
    cv_folds: Optional[int] = None,
) -> EvaluationSplit:
    """Build selection and evaluation splits."""
    if cv_folds is not None:
        if cv_folds < 2:
            raise ValueError(f"cv_folds must be at least 2 (got {cv_folds})")
        if len(samples) < cv_folds:
            raise ValueError(f"only {len(samples)} samples for {cv_folds} folds; "
                             "cannot split")
        folds = _stratified_folds(samples, cv_folds, seed)
        return EvaluationSplit(
            evaluation=list(samples), selection=list(samples),
            folds=[[s.sample_id for s in fold] for fold in folds])
    train = [s for s in samples if s.split == "train"]
    val = [s for s in samples if s.split == "val"]
    test = [s for s in samples if s.split == "test"]
    if test and (train or val):
        return EvaluationSplit(evaluation=list(test), selection=train + val)
    pool = train + val + test if (train or val or test) else list(samples)
    if not pool:
        return EvaluationSplit(evaluation=[], selection=[])
    rest, sel = split_train_val(pool, val_ratio=sel_ratio, seed=seed)
    return EvaluationSplit(evaluation=list(rest), selection=list(sel))

def split_train_val(
    samples: Sequence[RagSample],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[RagSample], List[RagSample]]:
    """Stratified train/validation split."""
    pos = [s for s in samples if s.label]
    neg = [s for s in samples if not s.label]
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_val_pos = max(1, int(round(len(pos) * val_ratio))) if pos else 0
    n_val_neg = max(1, int(round(len(neg) * val_ratio))) if neg else 0
    val = pos[:n_val_pos] + neg[:n_val_neg]
    train = pos[n_val_pos:] + neg[n_val_neg:]
    rng.shuffle(val)
    rng.shuffle(train)
    return train, val
