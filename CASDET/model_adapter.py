# -*- coding: utf-8 -*-

"""Model loading and access to Attention/FFN internal states.

ModelAdapter wraps a Hugging Face decoder-only model, captures the attention
weights and FFN outputs required by CASDET, and exposes the Logit Lens."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from .utils import get_device

logger = logging.getLogger("CASDET")

PathLike = str

@dataclass
class LayerStates:
    """Per-layer states for the response positions."""
    layer_idx: int
    attn_out: torch.Tensor
    e: torch.Tensor
    f: torch.Tensor
    x_mid: Optional[torch.Tensor] = None
    x: Optional[torch.Tensor] = None

class ModelAdapter:

    """Adapter for a decoder-only Hugging Face model."""
    SUPPORTED_ARCH = {"llama", "mistral", "qwen2"}

    def __init__(
        self,
        model_name_or_path: PathLike,
        device: str = "auto",
        dtype: str = "float16",
        trust_remote_code: bool = False,
    ) -> None:
        """Load the model and tokenizer."""
        self.model_name_or_path = str(model_name_or_path)
        self.multi_gpu = device == "auto_multi"
        self.device = (torch.device("cuda:0")
                       if self.multi_gpu else get_device(device))
        torch_dtype = {"float16": torch.float16,
                       "bfloat16": torch.bfloat16,
                       "float32": torch.float32}[dtype]
        load_kwargs: Dict[str, Any] = dict(
            attn_implementation="eager",
            device_map=("auto" if self.multi_gpu else self.device),
            trust_remote_code=trust_remote_code,
        )
        if self.multi_gpu:
            max_memory: Dict[int, int] = {}
            for i in range(torch.cuda.device_count()):
                free_bytes, _ = torch.cuda.mem_get_info(i)
                max_memory[i] = int(0.9 * free_bytes)
            load_kwargs["max_memory"] = max_memory
        import transformers.modeling_utils as _tmu  # noqa: F401
        if hasattr(_tmu, "check_torch_load_is_safe"):
            _tmu.check_torch_load_is_safe = lambda: None

        import transformers
        if int(transformers.__version__.split(".")[0]) >= 5:
            load_kwargs["dtype"] = torch_dtype
        else:
            load_kwargs["torch_dtype"] = torch_dtype
        try:
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, **load_kwargs)
        except ValueError as exc:
            if self.multi_gpu and "offloaded to the disk" in str(exc):
                n = torch.cuda.device_count()
                detail = []
                for i in range(n):
                    free_b, total_b = torch.cuda.mem_get_info(i)
                    detail.append(f"cuda:{i} free {free_b / 1e9:.1f} GB"
                                  f"/{total_b / 1e9:.1f} GB")
                raise RuntimeError(
                    f"Failed to load {model_name_or_path} with device_map='auto'. "
                    f"Visible GPUs: {n} "
                    f"({'; '.join(detail) if detail else 'no GPU'}), "
                    f"accelerate can only offload weights to disk.\n"
                    f"device=auto_multi requires enough free GPUs "
                    f"(Llama-2-13b in fp16 needs about 26 GB, more than one 24 GB GPU). "
                    f"Check CUDA_VISIBLE_DEVICES (a single visible GPU is incompatible "
                    f"with auto_multi) and whether another process occupies the GPUs. "
                    f"Setting offload_folder is not a fix; it runs the model on CPU."
                ) from exc
            raise
        self.model.eval()
        arch = self.model.config.model_type
        if arch not in self.SUPPORTED_ARCH:
            raise ValueError(
                f"Unsupported architecture: {arch}. CASDET supports Llama/Mistral-style "
                f"models.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._llm = self.model.model
        self.num_layers: int = len(self._llm.layers)
        self.hidden_size: int = self.model.config.hidden_size
        self.vocab_size: int = self.model.config.vocab_size
        self.num_heads: int = self.model.config.num_attention_heads

    @torch.no_grad()
    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return [T, V] logits from a standard forward pass."""
        out = self.model(input_ids=input_ids.unsqueeze(0).to(self.device),
                         use_cache=False)
        return out.logits[0].float().cpu()

    @contextmanager
    def capture(
        self,
        input_ids: List[int],
        resp_span: Tuple[int, int],
        ctx_mask: Sequence[int],
    ) -> Iterator[Dict[int, LayerStates]]:
        """Capture the states needed by CASDET for one input sequence."""
        ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        r_start, r_end = resp_span
        if r_start < 0 or r_end > len(input_ids) or r_end <= r_start:
            raise ValueError(
                f"invalid response span {resp_span} for sequence length {len(input_ids)}"
            )
        resp_rows = torch.arange(r_start, r_end, device=self.device)
        ctx_cols = torch.tensor(
            [i for i, m in enumerate(ctx_mask) if m],
            dtype=torch.long, device=self.device)

        store: Dict[int, LayerStates] = {}
        orig_forwards: List[Any] = []

        def make_attn_wrapper(li: int, attn: Any) -> Any:
            orig = attn.forward

            def wrapped(hidden_states: torch.Tensor, *args: Any, **kwargs: Any):
                kwargs = dict(kwargs)
                kwargs.pop("output_attentions", None)
                # Capture softmax attention, compute e_t^l, then release the weights.
                kwargs["output_attentions"] = True
                out = orig(hidden_states, *args, **kwargs)
                attn_output = out[0]
                attn_weights = out[1]
                # Under device_map="auto", this layer may live on a different GPU.
                dev = attn_weights.device
                rr = resp_rows.to(dev)
                cc = ctx_cols.to(dev)
                e = _context_contribution(
                    attn_weights=attn_weights,
                    hidden_states=hidden_states,
                    attn_module=attn,
                    resp_rows=rr,
                    ctx_cols=cc,
                )
                resp_out = attn_output[0][rr]
                store[li] = LayerStates(
                    layer_idx=li,
                    attn_out=resp_out.detach().clone(),
                    e=e.detach().clone(),
                    f=torch.zeros_like(resp_out),
                )
                if len(out) >= 3:
                    return (out[0], None, out[2])
                return (out[0], None)
            return orig, wrapped

        for li, layer in enumerate(self._llm.layers):
            orig, wrapped = make_attn_wrapper(li, layer.self_attn)
            layer.self_attn.forward = wrapped
            orig_forwards.append((layer.self_attn, orig))

        try:
            with torch.no_grad():
                self.model(input_ids=ids.unsqueeze(0), use_cache=False)
            self._finalize_x(store, ids, resp_rows)
            yield {li: store[li] for li in sorted(store)}
        finally:
            for module, orig in orig_forwards:
                module.forward = orig

    def _finalize_x(self, store: Dict[int, LayerStates],
                    ids: torch.Tensor, resp_rows: torch.Tensor) -> None:
        """Assemble the intermediate residual states from the captured layers."""
        dev = self.device
        embed = self._llm.embed_tokens(ids)[resp_rows].to(dev)
        prev_mid: Optional[torch.Tensor] = None
        prev_f: Optional[torch.Tensor] = None
        for li in range(self.num_layers):
            s = store[li]
            if li == 0:
                x_mid = embed + s.attn_out.to(dev)
            else:
                x_mid = prev_mid + prev_f + s.attn_out.to(dev)
            s.x_mid = x_mid
            s.f = self.run_ffn(li, x_mid).to(dev)
            s.x = x_mid + s.f
            prev_mid = x_mid
            prev_f = s.f

    @torch.no_grad()
    def run_ffn(self, layer_idx: int, x_mid_rows: torch.Tensor) -> torch.Tensor:
        """Run one FFN on the supplied intermediate states."""
        layer = self._llm.layers[layer_idx]
        layer_dev = layer.mlp.down_proj.weight.device
        h = layer.post_attention_layernorm(x_mid_rows.to(layer_dev))
        return layer.mlp(h)

    @torch.no_grad()
    def logit_lens(self, x: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits with the final norm and LM head."""
        norm = self._llm.norm
        W_U = self.model.lm_head.weight
        b = self.model.lm_head.bias
        return F.linear(norm(x.to(W_U.device)), W_U, b)

def _context_contribution(
    attn_weights: torch.Tensor,
    hidden_states: torch.Tensor,
    attn_module: Any,
    resp_rows: torch.Tensor,
    ctx_cols: torch.Tensor,
) -> torch.Tensor:
    """External-context contribution to the attention output (Eq. 4)."""
    if hidden_states.dim() == 3:
        hidden_states = hidden_states[0]
    A = attn_weights[0].float()
    H, Tq, Tk = A.shape
    hd = attn_module.head_dim
    n_heads = getattr(attn_module, "num_heads", None) or \
        attn_module.config.num_attention_heads
    n_kv = getattr(attn_module, "num_key_value_heads", None)
    if n_kv is None:
        groups = getattr(attn_module, "num_key_value_groups", None)
        n_kv = n_heads // groups if groups else n_heads
    assert Tq == hidden_states.shape[0], "attention rows and hidden states have different lengths"

    if ctx_cols.numel() == 0:
        return torch.zeros(resp_rows.numel(), attn_module.o_proj.weight.shape[0],
                           device=A.device, dtype=hidden_states.dtype)

    v = F.linear(hidden_states, attn_module.v_proj.weight,
                 attn_module.v_proj.bias)
    v = v.view(Tq, n_kv, hd)
    if n_kv < n_heads:
        v = v.repeat_interleave(n_heads // n_kv, dim=1)

    A_rc = A[:, resp_rows, :][:, :, ctx_cols]
    V_ctx = v[ctx_cols].float()
    s = torch.einsum("hrc,chd->rhd", A_rc, V_ctx)
    s = s.reshape(resp_rows.numel(), n_heads * hd)

    e = F.linear(s.to(attn_module.o_proj.weight.dtype),
                 attn_module.o_proj.weight, attn_module.o_proj.bias)
    return e.to(hidden_states.dtype)
