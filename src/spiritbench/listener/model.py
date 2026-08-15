import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HiddenStateModel:
    def __init__(self, model_id: str, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float16 if device == "mps" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, output_hidden_states=True).to(device).eval()
        self.n_layers = self.model.config.num_hidden_layers + 1

    @torch.no_grad()
    def _forward(self, text: str):
        ids = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.model(**ids)
        hs = torch.stack(out.hidden_states, dim=0)[:, 0]  # [n_layers, n_tokens, d]
        return hs.float().cpu().numpy(), out.logits[0, -1].float().cpu(), ids["input_ids"].shape[1]

    def hidden_states(self, text: str) -> np.ndarray:
        hs, _, _ = self._forward(text)
        return hs

    def hidden_states_with_spans(self, preamble: str, lines: list[str]):
        spans, prefix = [], preamble
        for i, line in enumerate(lines):
            start = len(self.tokenizer(prefix)["input_ids"])
            prefix = prefix + line + (".\n" if i < len(lines) - 1 else "")
            end = len(self.tokenizer(prefix)["input_ids"])
            spans.append((start, max(end, start + 1)))
        hs = self.hidden_states(prefix)
        spans = [(s, min(e, hs.shape[1])) for s, e in spans]
        return hs, spans

    @torch.no_grad()
    def yes_no_logprobs(self, prompt: str):
        _, logits, _ = self._forward(prompt)
        logprobs = torch.log_softmax(logits, dim=-1)

        def score(variants):
            tot = -np.inf
            for v in variants:
                toks = self.tokenizer(v, add_special_tokens=False)["input_ids"]
                if len(toks) == 1:
                    tot = np.logaddexp(tot, logprobs[toks[0]].item())
            return tot
        return score(["yes", " yes", "Yes", " Yes"]), score(["no", " no", "No", " No"])
