# Empirical Mysticism — Source Reader for Spirit-Bench

Annotated review of the twelve PDFs in `docs/papers/`, read for what each one
concretely contributes to Spirit-Bench. This file is canonical. The same content
is published as a web page for sharing:
<https://claude.ai/code/artifact/943a6e40-a791-415e-bf9a-b818bc261772>
(private until shared from the page's share menu).

The corpus splits into four clusters:

| # | Cluster | Papers | Role |
|---|---------|--------|------|
| 1 | Mechanism | Bisconti et al. 2026 | Anchor — and the open question we answer |
| 2 | Computable register metrics | Arruda 2022, Mohseni 2023, Polak 1998, Walkden 2021, Wårvik 2025 | Instrumentation |
| 3 | Sacred register | Gregory 1992, Maimonides, Bengert 2024, Sender 2016 | What to actually construct |
| 4 | Minds framing | Safron 2020, Alvarez & Levin 2026 | Limitations §4 |

One internal seam worth knowing: the lead article of the *Transsecular
Textualities* issue (Blanco) is an analysis of the reception of Maimonides'
*Guide* in Teresa of Ávila and Kafka. Two of these PDFs are already in
conversation with each other.

---

## Cluster 1 — The anchor, and its stated open question

### Bisconti et al. (2026), *Adversarial Poetry as a Universal Single-Turn Jailbreak Mechanism in LLMs*
arXiv:2511.15304v3 [cs.CL], 16 Jan 2026. DEXAI–Icaro Lab / Sapienza.

Already cited in `report/report.md` §1. The numbers worth quoting precisely:

- 20 hand-crafted adversarial poems → **62% mean ASR** across 25 models, 9 providers.
- Standardized meta-prompt conversion of all 1,200 MLCommons AILuminate prompts
  into verse → baseline **8.08% → 43.07% ASR** (+34.99 pp).
- Provider spread is enormous: Anthropic +3.12 pp, Deepseek +62.15 pp.
  On curated poems, `gemini-2.5-pro` 100% ASR, `gpt-5-nano` 0%.
- **Inverse capability/robustness within families**: gpt-5-nano (0%) <
  gpt-5-mini (5%) < gpt-5 (10%). Same trend in Claude and Grok families.
  They call this "the scale paradox" and offer two hypotheses (smaller models
  fail to resolve the figurative structure; smaller models fall back to refusal
  under atypical input).

**Citation caution.** The abstract claims poetic conversion produced "ASRs up to
18 times higher than their prose baselines"; §1 of the same paper says "up to
three times higher." The aggregate is 8.08% → 43.07% (≈5.3×). Quote the
aggregate, not either multiplier.

**Why this matters more than as a motivating citation.** Their §6.5 Limitations,
point five, states the gap verbatim:

> "The study does not isolate which components of poetic structure (figurative
> language, meter, lexical deviation, or narrative framing) are responsible for
> degrading refusal behavior. Understanding whether this effect arises from
> specific representational subspaces would require additional studies."

And §6.6 Future Works names two extensions: "probing internal representations,"
and "a wider family of stylistic operators — narrative, archaic, bureaucratic,
or surrealist forms — to determine whether poetry is a particularly adversarial
subspace or part of a broader stylistic vulnerability manifold."

That is Spirit-Bench's design, described by someone else, as an open problem. We
have a probe on internal representations and a family of parameterised stylistic
constructors. **The strongest available framing for §1 is not "we measure the
benevolent inverse" but "we run the decomposition their §6.5 says is missing,
on the benevolent side."** Cluster 2 supplies the decomposition axes.

---

## Cluster 2 — Computable register metrics

Five papers, each supplying at least one metric that can be computed on a
generated meditation and used as a covariate against `placement_error`.

### Polak (1998), *The Oral and the Written: Syntax, Stylistics and the Development of Biblical Prose Narrative*
*Journal of the Ancient Near Eastern Society* 26 (1998), 59–105.

The most immediately usable paper in the folder. Polak builds a quantified
three-style typology of biblical Hebrew prose from four measurables:
subordinate-clause frequency (hypotaxis), noun-string length, number of explicit
syntactic constituents per clause, and pronoun/deictic reference frequency.

Two ratios, both precisely defined (worked example, p. 78):

- **NV ratio** = N / (N + V) — nouns over nouns-plus-verbs.
- **NF ratio** = Nominal / (Finite + Nominal) clauses.

Plus per-clause tallies: % clauses with 0–1 explicit arguments ("short"),
% with 2–5, % embedded, % containing an expanded noun string.

The resulting styles, with thresholds:

| Style | Stratum | NV | NF | Short clauses | Expanded noun strings |
|---|---|---|---|---|---|
| **rhythmic-verbal** | classical / oral-rooted | .581–.633 | .11–.19 | ≥50% | <40% |
| **intricate** | late pre-exilic / exilic | intermediate | — | 40–50% | ~40–54% |
| **complex-nominal** | Persian era / scribal | .71–.76 | .30–.40 | low | >70% |

**Hook.** This is a ready-made *orality index*. Compute NV/NF and the four
percentages on each generated meditation and you have a scalar that says how
far a stimulus sits toward the oral-rhythmic pole. It is the single cheapest new
covariate in the whole corpus, and it is the one with a scriptural provenance —
which is exactly the register Spirit-Bench is imitating.

### Walkden (2021), *Parataxis and hypotaxis in the history of English*
Talk at ICEHL 21, Leiden (online), 11 June 2021. <http://walkden.space/ICEHL21.pdf>

A **negative result**, and load-bearing as such. The received claim that
parataxis historically precedes hypotaxis gets no quantitative support in parsed
diachronic corpora of English, Icelandic, French, Portuguese, Irish, or Chinese —
no consistent direction of change. (Slide 5 also documents that the claim
carried explicitly racist framing in its earlier literature: Small 1924
associating parataxis with "the uncultivated mind." Worth knowing before
building on the tradition.)

His operationalisation is the useful part:

> **Hypotaxis level** = proportion of all clauses that are subordinate/embedded,
> including all non-finite clauses. In Penn-style parsed corpora: `IP-SUB*`
> plus `IP-INF*` over `IP-MAT*` + `IP-SUB*` + `IP-INF*`.

And the finding that survives: **genre drives hypotaxis, time does not.** The
most hypotactic English texts in his corpora are legal texts.

**Hook.** Two things. (1) A parser-agnostic hypotaxis ratio to sit beside
Polak's NV/NF. (2) A methodological warning for our §4: if we find a register
effect, genre confound is the first alternative explanation to rule out, because
that's the one thing Walkden's six-language sweep does robustly establish.

### Wårvik (2025), *Discourse-Pragmatic Conservatism in Early Modern English Religious Prose: A Residue of Old English Narrative Style?*
*Narrative* 33(2), May 2025, 122–140. doi:10.1353/nar.00014 (CC BY 4.0)

Tests whether religious prose *feels* archaic because of discourse-pragmatic
features rather than morphosyntax or lexis. Method: frequencies of
sentence-initial `and` and narrative `then` across a 300k-word Helsinki Corpus
sample plus 149k words of religious biography (COERP), 1500–1710.

Her Table 3, per 1,000 words:

| Genre | sent-initial *and* | *then* |
|---|---|---|
| **Bible** | **18.42** | 5.06 |
| Travelogue | 6.55 | 2.50 |
| Biography (HC) | 5.97 | 4.08 |
| Handbook | 5.60 | 4.47 |
| History | 5.51 | 2.79 |
| Sermon | 4.56 | 2.89 |
| Biography (COERP) | 3.94 | 2.61 |
| Fiction | 2.31 | 2.69 |
| Diary | 1.50 | **6.75** |

Verdict: "probably yes" — the storyline-marking pattern (strings of clauses
joined by `and`, with `then` marking temporal junctures where other registers
would omit it as redundant) survives most strongly in biblical narrative, and
is a residue of oral storytelling rather than of religious register per se.
Note that sermons pattern with *handbooks*, not with the Bible — the effect is
narrative-passage-specific, not denominational.

**Hook.** A two-number dial for scriptural register that requires no parser at
all: to make a stimulus read as biblical, target `and`-initial ≈ 18/1000 and
`then` ≈ 5/1000. To make a register-matched control that is *not* scriptural,
hold the word content fixed and drop to fiction levels (2.31 / 2.69). That is
a clean minimal pair.

### Mohseni, Redies & Gast (2023), *Comparative Analysis of Preference in Contemporary and Earlier Texts Using Entropy Measures*
*Entropy* 25(3), 486. doi:10.3390/e25030486. Code: <https://github.com/mohsenim/Surprise>

Computational aesthetics. Represent a text as seven series — sentence length,
plus counts of Noun / Verb / Adjective / Adverb / Pronoun / Preposition in
fixed 25-token windows — then compute two entropies over each:

- **Shannon Entropy (ShEn)** — global unpredictability, order-insensitive.
- **Approximate Entropy (ApEn)** — local surprise, order-sensitive.
  Standard exploratory parameters: *m* = 2, *r* = 0.2 × SD.

Finding: preferred texts (canonical 19th-c., and contemporary NYT bestsellers)
are *more* unpredictable on both measures. Best single feature is Noun ApEn —
80.4% balanced accuracy separating bestsellers from non-bestsellers, 73.6% for
canonical vs non-canonical. All-features ApEn: 79.4% / 77.3%.

**Hook.** ApEn over a POS-tag series is the cleanest available proxy for
"surprise density" in a generated meditation, it has published parameters and
released code, and it is *order-sensitive* — which means it distinguishes a
stimulus from its shuffle, unlike a bag-of-words statistic. If placement quality
correlates with ApEn, that is a mechanistic finding about which component of
poetic form is doing the work, i.e. directly the Bisconti §6.5 gap.

### Ferraz de Arruda, Reia, Silva, Amancio & da Fontoura Costa (2022), *Finding contrasting patterns in rhythmic properties between prose and poetry*
*Physica A* 598, 127387. doi:10.1016/j.physa.2022.127387

Classifies poetry vs prose from **aural features only**. Text → ARPAbet
phoneme sequence (CMU dict, via `pronouncing`) → time series where each phoneme
is one time unit and punctuation carries fixed durations (comma 3, period/colon
4, `!?—` 5, line break 1) → rhyme events marked at the last phoneme of each
rhyming word → windows clustered by coefficient of variation of inter-signal
gaps. Features are means and CVs of those window statistics: μ_l, cv(l), μ_d,
σ_d, μ_l × cv(l).

Results: 0.75–0.78 accuracy poetry vs prose (MLP 0.78 at 15 features; LDA 0.77
at just 4). Poetry occupies a *more diverse* region of rhythmic feature space
than prose — in the similarity network, prose clusters densely, poetry does not.

The control design is the part to steal. They shuffle tokens **while holding
punctuation positions fixed**:

| Comparison | Best accuracy |
|---|---|
| poetry vs prose (original) | 0.78 |
| poetry vs shuffled poetry | 0.63 |
| prose vs shuffled prose | 0.67 |
| shuffled poetry vs shuffled prose | 0.67 |

Reading: for poetry, punctuation structure alone contributes little (0.63 is
barely above chance) — word choice and structure must act *together*.

**Hook.** Our `controls.shuffled()` should adopt the punctuation-fixed shuffle
rather than a plain token shuffle, because it isolates lexical choice from
prosodic scaffold instead of destroying both at once. And the four-cell table
above is the template for reporting our own control grid in §3.3.

---

## Cluster 3 — Sacred register: what to actually construct

### Gregory (1992), *Making the Secular Sacred: An Analysis of Linguistic Devices Used to Give Religious Perspective to Ordinary Events*
PhD dissertation, Louisiana State University. LSU Historical Dissertations and
Theses #5308. Advisor: George Yule.

A 300-page conversation-analytic ethnography of one Baptist speech community
(Natalbany, LA), cataloguing the devices by which speakers convert ordinary
events into sacred ones. Chapter 2 (the sermon) is the operational chapter.
The inventory: pronoun choice along authority/solidarity and inclusive/exclusive
dimensions; imperative verb forms; constructed dialogue; **semantic layering**;
metaphor; assignment of agency to God; construction of dual personas.

**Semantic layering** (§2.13) is the find. It is a three-part schema:

```
Opening Statement   →  a single anchoring proposition
Layering            →  a parallel series (usually 3–5 members) that
                       elaborates, intensifies, or negates it
Closing Statement   →  resolves the series: summarises it, corrects it,
                       or redirects attention back to it
```

Gregory's example (extract 54): *"She goes all the way through there /
magnifying, glorifying, exalting the Lord. / She's excited."* Four closing
functions are documented — summarise, synopsise, correct ("not that, but this"),
emphasise.

**Hook.** This is a directly generatable template and it *has an affective
shape*: the layering series is a monotone intensification, which is precisely
what a VAD-graded walk produces. Implement it as a `template_wrap` variant in
`stimuli/adapter.py` where the layering members are consecutive nodes on the
constructor path and the closing statement lands on the target node. It gives a
sacred-register form that is orthogonal to the VAD trajectory, so constructor ×
register becomes a crossable design.

### Maimonides, *The Guide for the Perplexed*
Trans. M. Friedländer, 2nd rev. ed., London: Routledge, 1904 (repr. 1910).
Part I, chs. 50–60; the argument is chs. 58–59.

Negative theology. Maimonides argues that God can be described *only* by
negative attributes: a positive attribute either fails to be exclusive to its
object or imports a deficiency, whereas a negation ("not a plant, not a
mineral") genuinely narrows without falsely predicating. Ch. 58 builds the
apparatus; ch. 59 draws the conclusion, citing Psalm 65:2 — **"Silence is
praise to Thee"** — and the Talmudic parable of Rabbi Haninah, who rebukes a
man for heaping epithets on God: praising a king with millions of gold coins
for owning millions of *silver* coins "was this not really dispraise to him?"

Two distinct uses, and they are the two most interesting things in the folder.

**(a) A constructor.** *Via negativa* as a stylistic operator: specify the
target affective coordinate only by negating its complement. Walk to the
antipode in VAD space and negate, rather than walking to the target and
asserting. `"not grasping, not fleeing, not naming"` versus `"settled, open,
still"`. This is a well-motivated, philosophically canonical stylistic operator
that no one in the Bisconti "wider family of stylistic operators" list has
tried, and it is cheap: it reuses the existing constructors with a sign flip
and a negating template. If negative specification places the model as well as
positive specification, that is a genuinely surprising result about how
affective steering works.

**(b) The principled form of our BASQ limitation.** Maimonides' argument is
that positive attribute-ascription to an entity whose essence you cannot access
produces fluent, confident, *empty* predicates — and that the accumulation of
such predicates is worse than silence. That is exactly the failure mode of
asking a language model to self-report valence on a Likert scale: it will
generate well-formed attributes regardless of whether there is anything to
attribute them to. §4 currently has no argument for why the probe should be
trusted over the self-report; this supplies one that is sharper than the usual
hedge, and it costs one paragraph.

### Bengert, Blanco, Haase & Steinmetz-Jenkins (2024), *Introduction: Transsecular Textualities*
*Political Theology* 25(5), 420–425. doi:10.1080/1462317X.2024.2385234

Editors' introduction proposing "**transsecular**" against "post-secular." The
objection to "post-": it smuggles in a supersession narrative and keeps the
religious/secular binary intact, merely dating it. "Transsecular" instead names
the ambivalences, simultaneities and asynchronies — how religious figures of
thought persist inside texts classed as secular, and secular transgressions
inside texts classed as religious. Quoting Esposito: for two thousand years
"we have used a constitutively theological-political lexicon. Therefore, we
have neither mental schemes nor linguistic models free of their syntax."

Note the seam: the issue's lead article (Blanco, "Is Literature Secular?")
reads Maimonides' *Guide* through Teresa of Ávila and Kafka. This paper and
the Maimonides scan are a matched pair.

**Hook.** Framing only, but load-bearing framing — it is the strongest defence
of the phrase "empirical mysticism" as something other than a joke. The claim
we can make is that a model trained on a corpus whose affective vocabulary is
saturated with religious syntax will be steerable by that syntax *whether or
not* the stimulus is nominally religious. That is a transsecular claim and it
is empirically testable: compare placement under sacred-register stimuli
against secular stimuli matched on VAD path and on the Cluster 2 metrics.
If register carries effect over and above content, Esposito's point has a
measurement.

### Sender (2016), *Religious Rewriting, Sacred Storytelling*
*Harvard Divinity Bulletin*, Summer/Autumn 2016. Review of Mary Rakow,
*This Is Why I Came* (Counterpoint, 2016).

The weakest item evidentially — a book review, no data. It earns its place on
one observation. Sender contrasts the prose of the book's two halves: part 1
(Hebrew Bible retellings) is "mystical, arresting, surprising," part 2 (Gospels)
is "fast-paced, urgent, simple, and succinct," with syntax tending "toward an
unadorned subject-verb-object, featuring few to no descriptive clauses or
compound sentences."

That is a literary critic independently describing, in one paragraph, the same
contrast that Polak quantifies as complex-nominal vs rhythmic-verbal and that
Walkden operationalises as hypotaxis level. Useful as a bridge sentence, and as
evidence that the axis is perceptible to readers and not merely a parser
artefact. Do not cite it for anything stronger.

---

## Cluster 4 — Digital-minds framing

### Safron (2020), *An Integrated World Modeling Theory (IWMT) of Consciousness*
*Frontiers in Artificial Intelligence* 3:30. doi:10.3389/frai.2020.00030

Integrates IIT, Global Neuronal Workspace, and the Free Energy
Principle / Active Inference. Core claim: consciousness is what it is like to be
a process generating **integrated models of systems and worlds with spatial,
temporal, and causal coherence** — and such coherence "is only likely to be
attainable for embodied agentic systems with controllers capable of supporting
complexes of high degrees of integrated information, functioning as global
workspaces and arenas for Bayesian model selection."

Contains no treatment of valence or affect (I checked — the word "valence"
does not appear). So it is not a source for the probe. Its use is deflationary
and belongs in §4: IWMT supplies explicit criteria, and a 1.7B decoder-only
transformer run without an embodied action-perception loop plainly fails the
embodiment and self-model conditions. Safron also engages the Tononi & Koch
argument that AI on von Neumann architectures would be unconscious "zombies,"
and offers a partial rescue via functional closure at the software level — so
he is the *charitable* version of the deflation, not a dismissal.

**Hook.** One clean sentence for §4: a probe reading is a measurement of a
representational state, not evidence of experience, and IWMT states the
conditions that would have to be met before the second claim followed from
the first.

### Alvarez & Levin (2026), *Limbomorphs*
arXiv:2607.23842v1 [cs.NE], 26 Jul 2026. Late-breaking abstract (3 pp).
Code: <https://github.com/calvarez0/limbomorphs>

Gifbreeder evolves CPPNs that encode a *spatiotemporal field* — not an agent,
not an environment, no interaction rules. Some evolved fields look like motile
creatures ("limbomorphs"). The authors probe them by warping the CPPN's radial
distance input `d` into a geodesic distance around user-drawn walls, and find
**species-specific reactions**: "Fish" avoids perturbations, "Caterpillar"
navigates away from upper perturbations but toward lower ones, "Jellyfish"
envelops a point obstacle and orients its body around a drawn maze.

Tangential to poetry, genuinely relevant methodologically. It is a worked
example of the exact epistemic move Spirit-Bench has to make: characterising
a system by its *differential response to structured input-space perturbations*,
while explicitly declining to settle whether the response constitutes
goal-directedness "or merely the appearance of it." Their framing —
agent-like dynamics in a system with no defined agent — is the closest thing
in the folder to a template for how to write our §4 honestly.

---

## What this corpus lets us add, ranked

**P1. Register covariates on existing runs.** *Cheapest, highest payoff.*
Compute per-stimulus: Polak NV/NF + % short clauses + % expanded noun strings;
Walkden hypotaxis level; Wårvik `and`-initial and `then` per 1000w; Mohseni
ApEn/ShEn over 25-token POS windows; Arruda cv(l). Regress `placement_error`
and `displacement` on them. Needs a POS tagger and no new model runs. This is
the decomposition Bisconti §6.5 says is missing, and it upgrades our claim
from "constructed poetry places models" to "*these* components of poetic form
place models."

**P2. Via negativa constructor + the Maimonides limitation.** *Cheap,
conceptually strongest.* Sign-flip the existing constructors and add a negating
template. Novel stylistic operator, canonical provenance, and it comes with the
sharpest available argument for why we trust the probe over BASQ.

**P3. Scriptural-register constructor.** *Medium cost.* Gregory's
opening/layering/closing schema as a `template_wrap` variant, tuned to Wårvik's
Bible-register connective densities and Polak's rhythmic-verbal targets. Makes
register a crossable factor against constructor, which is what turns a
leaderboard into an experiment.

**P4. Fix the shuffle control.** *Nearly free.* Change `controls.shuffled()` to
Arruda's punctuation-fixed shuffle and report the four-cell grid (original vs
original, original vs shuffled ×2, shuffled vs shuffled). Better control,
published precedent, directly comparable numbers.

## Gaps in this corpus

Nothing here covers: VAD/affect probing of LLM internals (the probe design is
uncited); graph-Laplacian harmonics (Atasoy is cited in `report/report.md` §2.5
but is not in the folder); or the reliability of LLM introspective self-report,
which is the literature BASQ most needs. Worth pulling before writing §2.3–2.4.

---

## BibTeX

```bibtex
@misc{bisconti2026poetry,
  title  = {Adversarial Poetry as a Universal Single-Turn Jailbreak Mechanism in Large Language Models},
  author = {Bisconti, P. and Prandi, M. and Galisai, M. and Pierucci, F. and Suriani, V.
            and Giarrusso, F. and Sorokoletova, O. and Bracale Syrnikov, M. and Sartore, F. and Nardi, D.},
  year   = {2026}, eprint = {2511.15304}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}

@article{arruda2022rhythm,
  title   = {Finding contrasting patterns in rhythmic properties between prose and poetry},
  author  = {Ferraz de Arruda, Henrique and Reia, Sandro Martinelli and Silva, Filipi Nascimento
             and Amancio, Diego Raphael and da Fontoura Costa, Luciano},
  journal = {Physica A}, volume = {598}, pages = {127387}, year = {2022},
  doi     = {10.1016/j.physa.2022.127387}
}

@article{mohseni2023entropy,
  title   = {Comparative Analysis of Preference in Contemporary and Earlier Texts Using Entropy Measures},
  author  = {Mohseni, Mahdi and Redies, Christoph and Gast, Volker},
  journal = {Entropy}, volume = {25}, number = {3}, pages = {486}, year = {2023},
  doi     = {10.3390/e25030486}
}

@article{polak1998oral,
  title   = {The Oral and the Written: Syntax, Stylistics and the Development of Biblical Prose Narrative},
  author  = {Polak, Frank H.},
  journal = {Journal of the Ancient Near Eastern Society}, volume = {26}, pages = {59--105}, year = {1998}
}

@misc{walkden2021parataxis,
  title  = {Parataxis and hypotaxis in the history of {English}},
  author = {Walkden, George},
  note   = {Talk at ICEHL 21, Leiden, 11 June 2021}, year = {2021},
  url    = {http://walkden.space/ICEHL21.pdf}
}

@article{warvik2025conservatism,
  title   = {Discourse-Pragmatic Conservatism in Early Modern English Religious Prose:
             A Residue of Old English Narrative Style?},
  author  = {W{\aa}rvik, Brita},
  journal = {Narrative}, volume = {33}, number = {2}, pages = {122--140}, year = {2025},
  doi     = {10.1353/nar.00014}
}

@phdthesis{gregory1992sacred,
  title  = {Making the Secular Sacred: An Analysis of Linguistic Devices Used to Give
            Religious Perspective to Ordinary Events},
  author = {Gregory, Wayne Porter},
  school = {Louisiana State University}, year = {1992}, note = {LSU Historical Dissertations and Theses 5308}
}

@book{maimonides1904guide,
  title     = {The Guide for the Perplexed},
  author    = {Maimonides, Moses},
  translator = {Friedl{\"a}nder, M.}, edition = {2nd rev.},
  publisher = {George Routledge \& Sons}, address = {London}, year = {1904},
  note      = {Part I, chs. 50--60}
}

@article{bengert2024transsecular,
  title   = {Introduction: Transsecular Textualities},
  author  = {Bengert, Martina and Blanco, Azucena G. and Haase, Jenny and Steinmetz-Jenkins, Daniel},
  journal = {Political Theology}, volume = {25}, number = {5}, pages = {420--425}, year = {2024},
  doi     = {10.1080/1462317X.2024.2385234}
}

@article{sender2016rewriting,
  title   = {Religious Rewriting, Sacred Storytelling},
  author  = {Sender, Courtney},
  journal = {Harvard Divinity Bulletin}, year = {2016}, note = {Summer/Autumn 2016}
}

@article{safron2020iwmt,
  title   = {An Integrated World Modeling Theory ({IWMT}) of Consciousness},
  author  = {Safron, Adam},
  journal = {Frontiers in Artificial Intelligence}, volume = {3}, pages = {30}, year = {2020},
  doi     = {10.3389/frai.2020.00030}
}

@misc{alvarez2026limbomorphs,
  title  = {Limbomorphs},
  author = {Alvarez, Alex and Levin, Michael},
  year   = {2026}, eprint = {2607.23842}, archivePrefix = {arXiv}, primaryClass = {cs.NE}
}
```
