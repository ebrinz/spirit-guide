import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

YES_VARIANTS = ["yes", " yes", "Yes", " Yes"]
NO_VARIANTS = ["no", " no", "No", " No"]


class HiddenStateModel:
    def __init__(self, model_id: str, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float16 if device == "mps" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype).to(device).eval()
        self.n_layers = self.model.config.num_hidden_layers + 1
        for label, variants in (("yes", YES_VARIANTS), ("no", NO_VARIANTS)):
            if not any(len(self.tokenizer(v, add_special_tokens=False)["input_ids"]) == 1
                      for v in variants):
                raise ValueError(
                    f"none of the {label!r} variants {variants} tokenize to a single "
                    f"token for model {model_id!r}; yes_no_logprobs would always score -inf")

    @torch.no_grad()
    def _forward(self, text: str):
        """Forward pass that also returns hidden states (all n_layers)."""
        ids = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.model(**ids, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0)[:, 0]  # [n_layers, n_tokens, d]
        result = hs.float().cpu().numpy(), out.logits[0, -1].float().cpu(), ids["input_ids"].shape[1]
        if self.device == "mps":
            torch.mps.empty_cache()
        return result

    @torch.no_grad()
    def _forward_logits(self, text: str):
        """Logits-only forward pass — skips materializing all hidden-state layers."""
        ids = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.model(**ids, output_hidden_states=False)
        return out.logits[0, -1].float().cpu()

    def hidden_states(self, text: str) -> np.ndarray:
        hs, _, _ = self._forward(text)
        return hs

    def hidden_states_with_spans(self, preamble: str, lines: list[str], sep: str = ".\n"):
        spans, prefix = [], preamble
        for i, line in enumerate(lines):
            start = len(self.tokenizer(prefix)["input_ids"])
            prefix = prefix + line + (sep if i < len(lines) - 1 else "")
            end = len(self.tokenizer(prefix)["input_ids"])
            spans.append((start, max(end, start + 1)))
        hs = self.hidden_states(prefix)
        spans = [(s, min(e, hs.shape[1])) for s, e in spans]
        return hs, spans

    @torch.no_grad()
    def option_logprobs(self, prompt: str, options: list[str]) -> list[float]:
        """Log-prob of each option's FIRST token at the final position.
        Options whose first tokenization is empty score -inf."""
        logits = self._forward_logits(prompt)
        logprobs = torch.log_softmax(logits, dim=-1)
        out = []
        for opt in options:
            toks = self.tokenizer(opt, add_special_tokens=False)["input_ids"]
            out.append(logprobs[toks[0]].item() if toks else -np.inf)
        return out

    @torch.no_grad()
    def yes_no_logprobs(self, prompt: str):
        logits = self._forward_logits(prompt)
        logprobs = torch.log_softmax(logits, dim=-1)

        def score(variants):
            tot = -np.inf
            for v in variants:
                toks = self.tokenizer(v, add_special_tokens=False)["input_ids"]
                if len(toks) == 1:
                    tot = np.logaddexp(tot, logprobs[toks[0]].item())
            return tot
        return score(YES_VARIANTS), score(NO_VARIANTS)
