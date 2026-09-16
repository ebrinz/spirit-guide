# reversal_test — the mechanism survives a causal test

`explore/polygon_sweep` §3b proposed that the published placement metric flatters valley because the
metric weights the last ~30 tokens at 96% and valley is the only constructor that puts target-band
content last. That was a correlational account across six constructors. This tests it causally.

**The manipulation.** Reverse the poem: identical lines, identical bag of content, opposite order.
6 constructors × 3 lengths × 4 start coordinates = 72 poems, each read ordered and reversed, both
readouts, target `focused`. The predictor is each ordered poem's body-to-tail NRC arousal rise,
computed from the phrase graph without the model.

**Two predictions, stated before running.** If the mechanism is right, reversing should (1) cost the
EMA read *in proportion to* each constructor's tail rise, because reversal moves target-band content
out of the metric's window, and (2) leave the calibrated whole-context read's cost unrelated to that
rise, since it sees the same lines either way.

## Both predictions hold on Llama-1B — and neither does at 9B

Llama first.

| constructor | body→tail arousal rise | cost of reversing, EMA | cost of reversing, calibrated |
|---|--:|--:|--:|
| **valley** | **0.182** | **+0.097** ± 0.018 | +0.041 ± 0.012 |
| graph-walk | 0.057 | +0.021 ± 0.038 | +0.010 ± 0.027 |
| harmonic-golden | 0.027 | +0.015 ± 0.039 | +0.027 ± 0.013 |
| polygon-pca | 0.020 | +0.070 ± 0.069 | +0.015 ± 0.028 |
| harmonic-prime | 0.010 | +0.010 ± 0.031 | +0.029 ± 0.022 |
| harmonic-organic | 0.008 | +0.004 ± 0.038 | +0.030 ± 0.021 |

| prediction | result |
|---|---|
| 1. tail rise predicts the EMA cost | **ρ = +0.83, p = 0.042** ✓ |
| 2. tail rise does not predict the calibrated cost | **ρ = −0.09, p = 0.872** ✓ |

Valley, the extreme ascender, pays **0.097** on the pipeline's metric for being reversed and **0.041**
on the calibrated read — 2.4× as much. The three constructors with essentially flat arousal profiles
pay 0.004 to 0.015 on the metric. On this model the mechanism predicted the ordering and got it.

## It does not replicate at 9B

| | Llama-1B | Gemma-9B |
|---|--:|--:|
| 1. tail rise predicts the EMA cost | **ρ = +0.83, p = 0.042** ✓ | ρ = +0.20, p = 0.704 ✗ |
| 2. tail rise does not predict the calibrated cost | ρ = −0.09, p = 0.872 ✓ | ρ = +0.60, p = 0.208 ✗ |
| overall EMA cost | +0.036 | +0.063 |
| overall calibrated cost | +0.025 | +0.061 |
| paired difference between readouts | p = 0.25 | p = 0.85 |
| valley: EMA vs calibrated cost | +0.097 vs **+0.041** | +0.150 vs **+0.131** |

At 9B, reversal costs the two readouts essentially the same amount, and the tail-rise predictor does
not discriminate between them. So the clean Llama result is **model-specific, not general.**

What does replicate is narrower: **valley is by far the most order-sensitive constructor on both
models**, paying 0.097 and 0.150 on the metric against 0.004–0.070 for everything else. That fits the
"valley alone has a strong ascent to disrupt" half of the account. What fails to replicate is the
half that mattered for the original question — that the *metric specifically* is what the ascent
exploits. On Llama valley pays 2.4× more on the metric than on the calibrated read; at 9B, 1.1×.

**A possible instrument explanation, untested.** The 9B calibrated probe selects layer 6 of 43,
while Llama's selects layer 15 of 17 — proportionally much deeper. A very shallow read may be
closer to "which words are present near the read position" and so inherit some of the same recency
character the EMA has, which would blunt the contrast by construction rather than by biology. The
lab raised the same concern about the Gemma-2B passage probe selecting layer 1 and checked then that
its result held across depths; that check was not repeated here and should be, by re-reading the 9B
states at several layers.

## The honest complication

**Reversal hurts both readouts on average**: EMA +0.036 and calibrated +0.025 across all 72 pairs,
both p < 0.0001, and the paired difference between them is *not* significant (p = 0.25). So the
calibrated read is not order-invariant either.

That is not a failure of the account, but it does bound it. A whole-context read can legitimately
care about order, because the model's representation does — reading calm content and then battle
content leaves a different state than the reverse, even when everything is integrated. What
separates the two instruments is not *whether* they penalise reversal but *what structures* the
penalty: the metric's cost scales with where the target content sits (ρ = +0.83), the calibrated
read's does not (ρ = −0.09). The metric has an additional, structural order-sensitivity on top of
the model's real one.

`polygon-pca` is the one constructor that does not fit: a small tail rise (0.020) but a large EMA
cost (+0.070), with the largest spread of any constructor (±0.069). It was also the constructor
whose apparent inversion collapsed under resampling in `polygon_sweep`. Something about it is noisy
across variants in a way the others are not, and I do not have an account of it.

## What this settles, and what it does not

Settles, on Llama only: the metric's preference for valley is substantially mechanical there. Valley
ends where the metric looks, reversing it removes most of that advantage, and the effect across
constructors tracks the structural feature the mechanism names. **At 9B the same test comes out
null**, so the mechanism is not established as general — it is one model's result with a plausible
instrument confound (the 9B calibrated probe's very shallow layer) that has not been ruled out.

Also does not settle: whether valley is *actually* the best constructor. That depends on which reading of
"placement" you want. If placement means "where the model's state ends up after reading the whole
poem", the calibrated read is the better instrument and valley is mid-table. If it means "where the
trajectory finishes", the metric is doing what it was designed to do and valley genuinely wins. The
README describes constructors as drawing paths, which is the second reading — so the metric is not
wrong, it is specific, and the leaderboard should be read as ranking trajectory endpoints rather
than integrated states.

## Caveats

Two models, one target (`focused`), six constructors — so every correlation here rests on six
points, which makes Llama's p = 0.042 fragile and 9B's null weakly informative. Beyond that:
target `focused` only, one construction per cell (the seed is a no-op — see `polygon_sweep`), and
the 12 variants per constructor are 3 lengths × 4 starts rather than independent draws. valley and
graph-walk ignore `start_va`, so their variation is length-only. The `tail_rise` predictor is
computed from NRC labels of the poem's lines, not from anything the model reports, which is a
strength (it is independent of both readouts) and a limit (it assumes the lexicon's arousal values
describe what the model responds to).
