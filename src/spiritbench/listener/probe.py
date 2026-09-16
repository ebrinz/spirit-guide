import pickle
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# High-dimensional states (d ≈ 2304) with a few thousand samples need
# standardization + strong regularization; alpha is selected per head.
ALPHA_GRID = [1e2, 1e3, 1e4]


@dataclass
class Probe:
    layer: int
    ridge_v: Ridge
    ridge_a: Ridge
    r2_v: float
    r2_a: float
    scaler: StandardScaler = field(default=None)
    # selection diagnostics; absent on probes pickled before these were added, so
    # read them with getattr(probe, name, None) rather than attribute access.
    layer_r2_v: np.ndarray = field(default=None)
    layer_rule: str = field(default=None)
    r2_se: float = field(default=None)

    def predict(self, hidden: np.ndarray) -> np.ndarray:
        X = hidden.astype(np.float64)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return np.stack([self.ridge_v.predict(X),
                         self.ridge_a.predict(X)], axis=1)


def collect_word_states(model, words, templates) -> np.ndarray:
    out = []
    for w in words:
        per_tmpl = [model.hidden_states(t.format(w=w))[:, -1, :] for t in templates]
        out.append(np.mean(per_tmpl, axis=0))  # [n_layers, d]
    return np.stack(out)


def _best_head(X_tr, X_te, y_tr, y_te, grid):
    best = (None, -np.inf)
    for alpha in grid:
        r = Ridge(alpha=alpha).fit(X_tr, y_tr)
        r2 = r2_score(y_te, r.predict(X_te))
        if r2 > best[1]:
            best = (r, r2)
    return best


def _r2_se(y_true, y_pred, n_boot=400, seed=0) -> float:
    """Bootstrap standard error of held-out R², resampling the test set."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    if n < 8:
        return 0.0
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        if np.ptp(y_true[idx]) == 0:          # degenerate resample
            vals[b] = np.nan
            continue
        vals[b] = r2_score(y_true[idx], y_pred[idx])
    return float(np.nanstd(vals))


def train_probe(states, v_targets, a_targets, alpha, test_frac, seed=0,
                layer_rule="one_se_deepest", n_boot=400) -> Probe:
    """Standardized ridge per layer, alpha selected from ALPHA_GRID per head.

    Layer selection (`layer_rule`):

    - ``"one_se_deepest"`` (default) — the one-standard-error rule. Among layers whose held-out
      valence R² is within one bootstrap SE of the best layer's, take the **deepest**. The R² curve
      over layers is often flat within noise (on gemma-2-9b it spans 0.886–0.932 over 43 layers),
      and a bare argmax then picks an arbitrary layer that can be very shallow. Shallow reads carry
      recency/lexical character rather than integrated state: selecting layer 6 of 43 for the 9B
      passage probe this way silently reversed a conclusion about order-sensitivity
      (`explore/reversal_test/LAYER_CHECK.md`). Preferring depth among statistically tied layers
      chooses the most integrated read consistent with the data.
    - ``"argmax"`` — plain argmax of held-out valence R², the behaviour before this rule existed.
      Use it to reproduce probes and results built earlier.

    `alpha` is kept for API stability — it joins ALPHA_GRID if not already present.
    """
    if layer_rule not in ("one_se_deepest", "argmax"):
        raise ValueError(f"unknown layer_rule {layer_rule!r}")
    grid = sorted({alpha, *ALPHA_GRID})
    n_layers = states.shape[1]
    idx_tr, idx_te = train_test_split(np.arange(len(states)), test_size=test_frac,
                                      random_state=seed)
    y_v_te, y_a_te = v_targets[idx_te], a_targets[idx_te]
    fits, r2s = [], np.empty(n_layers)
    for layer in range(n_layers):
        X_tr = states[idx_tr, layer].astype(np.float64)
        X_te = states[idx_te, layer].astype(np.float64)
        sc = StandardScaler().fit(X_tr)
        X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)
        rv, r2v = _best_head(X_tr, X_te, v_targets[idx_tr], y_v_te, grid)
        ra, r2a = _best_head(X_tr, X_te, a_targets[idx_tr], y_a_te, grid)
        fits.append((rv, ra, r2v, r2a, sc, X_te))
        r2s[layer] = r2v

    top = int(np.argmax(r2s))
    if layer_rule == "argmax":
        chosen, se = top, None
    else:
        # SE of the best layer's R², then the deepest layer within one SE of it
        rv_top, _, _, _, _, X_te_top = fits[top]
        se = _r2_se(y_v_te, rv_top.predict(X_te_top), n_boot=n_boot, seed=seed)
        tied = np.where(r2s >= r2s[top] - se)[0]
        chosen = int(tied.max())

    rv, ra, r2v, r2a, sc, _ = fits[chosen]
    return Probe(chosen, rv, ra, r2v, r2a, scaler=sc,
                 layer_r2_v=r2s, layer_rule=layer_rule, r2_se=se)


def save_probe(probe, path):
    with open(path, "wb") as f:
        pickle.dump(probe, f)


def load_probe(path) -> Probe:
    # Safe: probe.pkl is generated by this codebase, not from untrusted sources
    with open(path, "rb") as f:
        return pickle.load(f)
