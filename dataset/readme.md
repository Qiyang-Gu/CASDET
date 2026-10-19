# Dataset preparation

The datasets are **not** included in this repository. Download them from the
official sources below and place the files as shown. All paths are relative to
the repository root.

## Expected layout

```
dataset/
├── ragtruth/
│   ├── redeep_raw/
│   │   ├── response.jsonl
│   │   ├── response_with_llama3_8b.jsonl
│   │   └── source_info.jsonl
│   ├── llama2_7b/response.jsonl
│   ├── llama2_13b/response.jsonl
│   ├── llama3_8b/response.jsonl
│   └── mistral_7b/response.jsonl
└── hallurag/
    ├── {train,val,test}_meta-llama_Llama-2-7b-chat-hf.pickle
    ├── {train,val,test}_meta-llama_Llama-2-13b-chat-hf (int8).pickle
    ├── {train,val,test}_mistralai_Mistral-7B-Instruct-v0.1.pickle
    ├── llama2_7b.json
    ├── llama2_13b.json
    └── mistral_7b.json
```

## RAGTruth

The raw responses and annotations come from RAGTruth:

- Paper: Niu et al., *RAGTruth: A Hallucination Corpus for Developing
  Trustworthy Retrieval-Augmented Language Models*, ACL 2024.
- Official repository: <https://github.com/ParticleMedia/RAGTruth>

The Llama3-8B extension and the three JSONL files used as `redeep_raw/` are
the processed release distributed with ReDeEP (Sun et al., ICLR 2025):
<https://github.com/Jeryi-Sun/ReDEeP-ICLR> (see the dataset link in its
README). They are used here as data only.

Place the following three files in `dataset/ragtruth/redeep_raw/`:

| File | Size | Used by |
|---|---:|---|
| `response.jsonl` | 21.5 MB | Llama2-7B / Llama2-13B / Mistral-7B responses |
| `response_with_llama3_8b.jsonl` | 23.8 MB | adds the Llama3-8B extension |
| `source_info.jsonl` | 15.1 MB | prompts, task type and retrieved context |

Then generate the per-model files:

```bash
python scripts/prepare_ragtruth.py \
    --raw-dir dataset/ragtruth/redeep_raw \
    --out-dir dataset/ragtruth
```

Expected outputs (all UTF-8 JSONL):

| File | Records | Split | MD5 prefix |
|---|---:|---|---|
| `llama2_7b/response.jsonl` | 2965 | train 2515 / test 450 | `90145567e2905b2c` |
| `llama2_13b/response.jsonl` | 2965 | train 2515 / test 450 | `c56ab67fc587abf7` |
| `mistral_7b/response.jsonl` | 2965 | train 2515 / test 450 | `774247f10dc8f2ae` |
| `llama3_8b/response.jsonl` | 450 | test 450 (20% used for selection) | `736b200a86894287` |

Evaluation protocol: responses marked `quality != good` (official
`incorrect_refusal` and `truncated`) are filtered out at load time. These
responses contain no unsupported content and are not labelled as
hallucinations; keeping them would turn the task into refusal detection. The
same rule is applied to every evaluated method. Set `quality: null` in a
configuration file to disable the filter (sensitivity analysis only).

## HalluRAG

- Paper: Ridder & Schilling, *HalluRAG: Detecting Hallucinations in RAG
  Applications Using an LLM's Internal States*, 2025.
- DOI: <https://doi.org/10.17879/84958668505>
- License: CC BY-SA 4.0 (see the `LICENSE.txt` distributed with the dataset).

Download the official archive, extract the pickle files into
`dataset/hallurag/`, and keep the following **nine** files (the other
precision variants are not used):

- `{train,val,test}_meta-llama_Llama-2-7b-chat-hf.pickle` (unquantized)
- `{train,val,test}_meta-llama_Llama-2-13b-chat-hf (int8).pickle`
- `{train,val,test}_mistralai_Mistral-7B-Instruct-v0.1.pickle` (unquantized)

Then convert them:

```bash
python scripts/prepare_hallurag.py --data-dir dataset/hallurag --quant-13b int8
```

Expected outputs:

| File | Records | MD5 prefix |
|---|---:|---|
| `llama2_7b.json` | 1080 | `f7d4043b73ea9db8` |
| `llama2_13b.json` | 1080 | `b3466a33f143b24e` |
| `mistral_7b.json` | 1072 | `263636f4fd16f7a5` |

The loader keeps answerable instances only and evaluates all of them with
5-fold cross-validation. Llama3-8B is not reported because HalluRAG does not
provide responses for it.

## Licensing

Please follow the license of each dataset. Do not redistribute raw data with
this repository. The code license in the repository root does not apply to the
datasets or to the model weights.
