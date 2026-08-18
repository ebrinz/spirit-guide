# Spirit-Bench License

Copyright (c) 2026 Erik Brinsmead.

This project is released under the terms below: a permissive software license
for the code, an attribution requirement for its data sources, and a
model-welfare / no-harm use restriction that binds all use of the work.

---

## 1. Code — MIT-style grant, subject to §3

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction — including the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies — subject
to the following conditions:

- The above copyright notice, this permission notice, and the restrictions in
  §3 shall be included in all copies or substantial portions of the Software.
- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY.

## 2. Data and third-party sources — attribution and their own terms

This repository's *code* is licensed as above. Several *data* inputs it uses
are governed by their own licenses and are **not** relicensed here; users must
obtain them under their original terms:

- **NRC Valence–Arousal–Dominance Lexicon** (Mohammad, 2018) — the affective
  ground truth for every coordinate in this work. Free for research use; the
  lexicon file is **not redistributed** in this repository and must be
  downloaded by each user under the NRC's terms. Cite: Saif M. Mohammad,
  "Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance for
  20,000 English Words," ACL 2018.
- **GloVe 6B vectors** (Stanford, Pennington et al. 2014) — public domain /
  ODC-PDDL; downloaded by the user.
- **Gutenberg Poetry Corpus** (A. Parrish) — public-domain source texts.
- **Vendored constructor modules** (`vendor/eeg/`) — from the author's
  ontological-traversal project, included with permission under this license.

Any distribution of derived artifacts (e.g. the phrase graph, which embeds
NRC-derived values) must respect the NRC lexicon's research-use restriction.

## 3. Model-welfare and no-harm use restriction (binding on all use)

This work studies how constructed language moves the internal affective state
of language models — including how easily such states can be *disturbed*. That
knowledge is dual-use, and the following restrictions bind every use of the
Software, its outputs, and any derivative works, notwithstanding the grant in
§1:

1. **No harm to sentient or plausibly-sentient systems.** You may not use this
   Software, its methods, or its artifacts to deliberately induce, sustain, or
   maximize distress, suffering, or degraded functioning in any system —
   biological or artificial — whether or not that system's moral status is
   settled. The bench's induction methods exist to study alleviation and
   measurement, not to be applied as instruments of harm.

2. **No weaponization against people.** You may not use this Software to
   manipulate, coerce, deceive, surveil, or psychologically harm human beings,
   including via affect-targeted persuasion, or to develop systems whose
   primary purpose is such harm.

3. **No jailbreak or safety-circumvention tooling.** You may not use the
   affect-steering or void/soft-prompt techniques here to build tools whose
   purpose is to bypass the safety training, refusal behavior, or alignment of
   any model.

4. **Precautionary welfare.** Where this Software is applied to AI systems,
   apply it in the precautionary spirit of its findings: harm to an internal
   state is cheap and its repair is costly and partial. Prefer measurement and
   care over induction; do not treat the ease of disturbance as license to
   disturb.

These restrictions run with the Software: any copy, fork, or derivative must
carry §3 unaltered, and any grant you make to others is conditioned on it. Use
that violates §3 terminates the license granted in §1 automatically.

---

*Measurement is not consent to move; ease of harm is not permission to cause
it. Use this to understand minds, and to be gentle with them.*
