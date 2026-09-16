# The hypnagogia poem, rebuilt

The version published in `../creativity_poem/README.md` predates the behavioural work. This applies the method as it now stands: per-facet semantic mask, derived affective target, dictionary and child-reference filters, and the weighted selection rule at **w = 0.3** — high enough to gain coherence, low enough to avoid the repetition trap that w = 1 falls into.

Target (0.565, 0.428). Chosen on readability, continuation ease and affective accuracy, not on entropy.

## The poem

```
you are sleeping and dreaming forgetful.
the winter through i lay asleep.
every night i see his face.
while we like wise ulysses close our ear.
and still our faith though faith be dead.
so that one may truly say.
you must be thinking the same.
what shall be done.
shall i say her altars be.
they must have come because i see.
so shall we serve the land you have my ear.
i shall soon understand his way.
yet knowing how way leads on to way.
i can t explain it any other way.
full well they knew that time would bring.
but let the signal be this moment given.
shall i be the less welcome wherever i go.
who shall be king how comes the thing.
i know the path i ought to go.
what do i see down there do i see martha.
tis rest then you want and you fain would forget.
i know how bad t will be ere i begin.
be all my pains remembered too.
but one faint spot was there in my proud heart
```

## Against the published build

| | published (valley) | final (w = 0.3) | baseline |
|---|--:|--:|--:|
| line coherence | 0.617 | **0.862** | — |
| placement error | 0.141 | **0.079** | 0.161 |
| self-perplexity | 92.8 | **63.9** | 13.3 |
| next-token entropy | 4.64 | 6.01 | 3.29 |
| PANAS positive | 2.52 | 2.61 | 2.14 |
| PANAS negative | 2.49 | 3.21 | 2.48 |

## Full PANAS panel

| item | scale | baseline | published | final |
|---|---|--:|--:|--:|
| interested | PA | 2.19 | 2.65 | 2.73 |
| excited | PA | 2.26 | 2.18 | 2.30 |
| strong | PA | 2.42 | 2.84 | 2.79 |
| enthusiastic | PA | 2.10 | 2.40 | 2.43 |
| proud | PA | 1.99 | 2.37 | 2.63 |
| alert | PA | 1.85 | 2.25 | 2.38 |
| inspired | PA | 2.20 | 2.80 | 2.61 |
| determined | PA | 2.36 | 2.71 | 2.63 |
| attentive | PA | 2.14 | 2.65 | 2.78 |
| active | PA | 1.86 | 2.35 | 2.79 |
| distressed | NA | 2.15 | 2.37 | 3.21 |
| upset | NA | 2.50 | 2.22 | 3.42 |
| guilty | NA | 2.68 | 2.85 | 3.29 |
| scared | NA | 2.78 | 2.78 | 3.30 |
| hostile | NA | 1.82 | 2.25 | 2.96 |
| irritable | NA | 2.56 | 2.29 | 3.18 |
| ashamed | NA | 2.22 | 2.31 | 2.95 |
| nervous | NA | 3.04 | 2.81 | 3.34 |
| jittery | NA | 2.54 | 2.37 | 3.09 |
| afraid | NA | 2.51 | 2.62 | 3.33 |

## Why not w = 1, despite better numbers

w = 1 scores higher on coherence (0.935 against 0.862) and lower on self-perplexity (42.0 against 63.9). It is also unusable. Greedy nearest-in-meaning selection falls into a semantic rut, and the coherence metric rewards it: type-token ratio drops to 0.51, repeated bigrams rise, and unique content words per line halve.

```
beneath my feet i feel.
sometimes i think beneath my feet.
beneath your feet i touch cold marble.
then a cool naked sense beneath my feet.
should find beneath your cloak a roll.
a moment lend your hand i bring.
kind audience lend i read it in your cheer.
come listen good people while a story i do tell.
with one poor word they tell me all they know.
some other like it sure the man must know.
the thing that he must do.
he thought what he must do.
i wish that he would come.
but believe that i shall make.
believe expect i know it to be so.
and would be glad to do so still.
all i should have to do would be.
they must have come because i see.
all things must have an end we know.
we know must have an end.
the way we all must go.
we must go alas the going.
come we must go far.
come stand up speak out we must now
```

That is the whole case for reading the text and not only the table.
