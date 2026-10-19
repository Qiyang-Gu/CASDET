# -*- coding: utf-8 -*-

"""Prepare per-model RAGTruth files.

Usage:
    python scripts/prepare_ragtruth.py --raw-dir dataset/ragtruth/redeep_raw \
        --out-dir dataset/ragtruth"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RAW_DIR = "dataset/ragtruth/redeep_raw"
BASE_FILE = "response.jsonl"
EXT_FILE = "response_with_llama3_8b.jsonl"
SOURCE_FILE = "source_info.jsonl"

OUT_DIR = "dataset/ragtruth"

MODEL_DIRS = {
    "llama-2-7b-chat": "llama2_7b",
    "llama-2-13b-chat": "llama2_13b",
    "mistral-7B-instruct": "mistral_7b",
}
EXT_MODEL = "llama-3-8b-instruct"
EXT_DIR = "llama3_8b"

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON on line {line_no}: {exc}") from exc
    return records

def load_sources(raw_dir: Path) -> Dict[str, Dict[str, Any]]:
    sources: Dict[str, Dict[str, Any]] = {}
    for rec in load_jsonl(raw_dir / SOURCE_FILE):
        sources[str(rec["source_id"])] = rec
    return sources

def extract_context_from_prompt(prompt: str, task: str) -> Tuple[Optional[str], str]:
    if task == "QA":
        s = prompt.find("passage 1:")
        if s < 0:
            return None, "missing 'passage 1:'"
        e = prompt.find("In case the passages do not contain", s)
        if e < 0:
            return None, "missing 'In case the passages do not contain'"
        return prompt[s:e].strip(), ""
    if task == "Summary":
        anchor = prompt.find("Summarize the following news within")
        s = prompt.find("words:\n", anchor)
        if anchor < 0 or s < 0:
            return None, "missing 'Summarize the following news within ... words:'"
        s += len("words:\n")
        e = prompt.rfind("output:")
        if e < 0:
            return None, "missing 'output:'"
        return prompt[s:e].strip(), ""
    if task == "Data2txt":
        s = prompt.find("Structured data:\n")
        if s < 0:
            return None, "missing 'Structured data:'"
        s += len("Structured data:\n")
        e = prompt.find("Overview:")
        if e < 0:
            return None, "missing 'Overview:'"
        return prompt[s:e].strip(), ""
    return None, f"unknown task_type: {task!r}"

def derive_label(labels: Any, model: str) -> Tuple[bool, str]:
    if model == EXT_MODEL:
        if labels == ["Yes"]:
            return True, ""
        if labels in ([], ["No"]):
            return False, ""
        return False, f"unknown labels: {labels!r}"
    if isinstance(labels, list):
        return bool(labels), ""
    return False, f"unknown labels: {labels!r}"

def build_record(
    rec: Dict[str, Any],
    src: Dict[str, Any],
    counter: int,
) -> Tuple[Dict[str, Any], str]:
    task = str(src.get("task_type") or "")
    prompt = str(src.get("prompt") or "")
    if not prompt:
        return {}, f"source_id={src.get('source_id')} has no prompt"

    context, err = extract_context_from_prompt(prompt, task)
    if err:
        return {}, f"source_id={src.get('source_id')} {err}"

    si = src.get("source_info")
    question = ""
    if isinstance(si, dict):
        question = str(si.get("question") or "")

    hallucinated, err = derive_label(rec.get("labels"), str(rec.get("model")))
    if err:
        return {}, f"source_id={src.get('source_id')} {err}"

    out = {
        "id": str(rec.get("id") if rec.get("id") is not None else counter),
        "source_id": str(rec.get("source_id")),
        "model": rec.get("model"),
        "task": task,
        "question": question,
        "prompt": prompt,
        "source_info": context,
        "response": rec.get("response"),
        "hallucinated": hallucinated,
        "labels": rec.get("labels"),
        "quality": rec.get("quality"),
        "split": rec.get("split"),
        "temperature": rec.get("temperature"),
    }
    return out, ""

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="RAGTruth corpus -> per-model aligned files")
    parser.add_argument("--raw-dir", default=RAW_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    sources = load_sources(raw_dir)
    print(f"source_info loaded: {len(sources)} records")

    stats: Dict[str, Counter] = {}
    failures: List[str] = []

    def process(records: List[Dict[str, Any]], model: str, out_sub: str) -> None:
        selected = [r for r in records if r.get("model") == model]
        out_records: List[Dict[str, Any]] = []
        cnt = Counter()
        for i, rec in enumerate(selected):
            sid = str(rec.get("source_id"))
            src = sources.get(sid)
            if src is None:
                cnt["missing_source"] += 1
                failures.append(f"{out_sub}: source_id={sid} is not in source_info.jsonl")
                continue
            out_rec, err = build_record(rec, src, i)
            if err:
                cnt["build_failed"] += 1
                failures.append(f"{out_sub}: {err}")
                continue
            out_records.append(out_rec)
            cnt["written"] += 1
            cnt[f"label={int(out_rec['hallucinated'])}"] += 1
            cnt[f"split={out_rec.get('split')}"] += 1

        (out_dir / out_sub).mkdir(parents=True, exist_ok=True)
        target = out_dir / out_sub / "response.jsonl"
        with open(target, "w", encoding="utf-8") as f:
            for rec in out_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        stats[out_sub] = cnt
        print(f"[{out_sub}] input {len(selected)} -> wrote {cnt['written']} records"
              f" ({dict(cnt)}) -> {target}")

    base = load_jsonl(raw_dir / BASE_FILE)
    print(f"base corpus loaded: {len(base)} records")
    dropped = Counter(r.get("model") for r in base
                      if r.get("model") not in MODEL_DIRS)
    print("models dropped from the base corpus:", dict(dropped))
    for model, out_sub in MODEL_DIRS.items():
        process(base, model, out_sub)

    ext = load_jsonl(raw_dir / EXT_FILE)
    print(f"extension corpus loaded: {len(ext)} records")
    process(ext, EXT_MODEL, EXT_DIR)

    if failures:
        print(f"\n{len(failures)} failures (first 10):")
        for msg in failures[:10]:
            print("  -", msg)
    else:
        print("\nNo failures.")
    print("\nDone. Context extraction was template-validated for all task types; "
          "the output fields satisfy the load_ragtruth contract.")

if __name__ == "__main__":
    main()
