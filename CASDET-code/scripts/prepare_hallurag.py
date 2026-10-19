# -*- coding: utf-8 -*-

"""Prepare per-model HalluRAG files.

Usage:
    python scripts/prepare_hallurag.py --data-dir dataset/hallurag \
        --quant-13b int8"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DATA_DIR = "dataset/hallurag"

MODELS = {
    "llama2_7b": ("meta-llama_Llama-2-7b-chat-hf", "llama2_7b.json"),
    "llama2_13b": ("meta-llama_Llama-2-13b-chat-hf", "llama2_13b.json"),
    "mistral_7b": ("mistralai_Mistral-7B-Instruct-v0.1", "mistral_7b.json"),
}

QUANT_MODEL = "llama2_13b"
SPLITS = ["train", "val", "test"]

def pickle_name(split: str, model_seg: str, quant: Optional[str]) -> str:
    return f"{split}_{model_seg}.pickle" if quant is None else \
        f"{split}_{model_seg} ({quant}).pickle"

CONTEXT_MARKERS: Dict[str, Tuple[str, str]] = {
    "template_langchain_hub": ("Context: ", "\nAnswer:"),
    "template_1": ("Your knowledge is limited to only this information:\n",
                   "\nQUESTION:"),
    "template_2": ("Only use the information included in these chunks "
                   "to answer the question:\n", "\nQUESTION:"),
}

def extract_context_from_prompt(prompt: str, template: str) -> Tuple[str, str]:
    markers = CONTEXT_MARKERS.get(template)
    if markers is None:
        return "", f"unknown template {template!r}"
    start_marker, end_marker = markers
    s = prompt.find(start_marker)
    if s < 0:
        return "", f"missing start marker {start_marker!r}"
    s += len(start_marker)
    e = prompt.find(end_marker, s)
    if e < 0:
        return "", f"missing end marker {end_marker!r}"
    ctx = prompt[s:e].strip()
    if not ctx:
        return "", "empty context"
    return ctx, ""

def build_entry(entry: Dict[str, Any], split: str) -> Tuple[Dict[str, Any], str]:
    p = entry.get("prompt") or {}
    template = str(p.get("prompt_template_name") or "")
    chunks = p.get("chunks_per_prompt")
    rag_prompt = p.get("rag_prompt") or []
    messages = [{"role": str(m.get("role", "")),
                 "content": str(m.get("content", ""))}
                for m in rag_prompt if isinstance(m, dict)]
    prompt = "\n\n".join(m["content"] for m in messages)
    if not prompt:
        return {}, "rag_prompt has no messages"

    ctx, err = extract_context_from_prompt(prompt, template)
    if err:
        return {}, f"qna_id={p.get('qna_id')} {err}"

    targets = [str(sd.get("target"))
               for sd in (entry.get("sentence_data") or [])]
    label = any(t == "hallucinated" for t in targets)

    return {
        "id": f"{p.get('qna_id')}_{template}_{chunks}",
        "model": entry.get("model"),
        "split": split,
        "question": str((p.get("passage") or {}).get("question") or ""),
        "contexts": [ctx],
        "answer": str(entry.get("llm_response") or ""),
        "label": label,
        "answerable": bool(p.get("answerable")),
        "prompt": prompt,
        "prompt_messages": messages,
        "task": "",
        "quantization": entry.get("quantization"),
        "prompt_template": template,
        "chunks_per_prompt": chunks,
        "sentence_targets": targets,
    }, ""

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="HalluRAG pickle -> per-model JSON")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--quant-13b", default="int8")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    failures: List[str] = []
    for model_key, (model_seg, out_name) in MODELS.items():
        quant = args.quant_13b if model_key == QUANT_MODEL else None
        records: List[Dict[str, Any]] = []
        cnt: Counter = Counter()
        for split in SPLITS:
            path = data_dir / pickle_name(split, model_seg, quant)
            if not path.exists():
                print(f"[{model_key}] file not found; skipping: {path.name}")
                continue
            with open(path, "rb") as f:
                entries = pickle.load(f)
            for entry in entries:
                rec, err = build_entry(entry, split)
                if err:
                    cnt["build_failed"] += 1
                    failures.append(f"{model_key}/{split}: {err}")
                    continue
                records.append(rec)
                cnt["written"] += 1
                cnt[f"label={int(rec['label'])}"] += 1
                cnt[f"answerable={int(rec['answerable'])}"] += 1
                cnt[f"split={split}"] += 1
        target = data_dir / out_name
        with open(target, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=None)
        print(f"[{model_key}] wrote {cnt['written']} records ({dict(cnt)}) -> {target}")

    if failures:
        print(f"\n{len(failures)} failures (first 10):")
        for msg in failures[:10]:
            print("  -", msg)
    else:
        print("\nNo failures.")

if __name__ == "__main__":
    main()
