# A poem for creativity and mentation

Target **(0.723, 0.471)** — the NRC centroid of 49 words across making, mentation, insight, curiosity and imagination. The arousal is mid-range on purpose: every experiment in `explore/` found it capped near 0.5, so 0.85 would be a target nothing reaches.

## The one to use

**valley, 24 lines, clean+semantic** — biggest movement on the self-report items that bear on mentation (inspired +0.96, attentive +0.75) with placement close behind the best (0.094).

```
i thought it might be pan.
in the books you have read.
to be honest.
for soul can be what soul hath been.
what spiritual meanings gird like air the earth.
o the pleasant sight to see.
think i might be looking blue.
we know we dream we dream we know.
let whoso can before such praying books.
for nature s sad reality.
yet in the go cart patience give it time.
if those dim eyes can yet ulysses know.
avoid to touch them and be wise.
what would i give to know.
shall bring forth fruit this muse shall speak to thee.
as one could wish to know.
knot themselves to make her trip.
but my heart will leap at a scene like this.
great guide i ask you still.
a supernatural faith to paint this christ.
pause take thy choice each gem a host can buy.
kindness to any one to show.
how much happier would i be.
sometimes whoever seeks abroad may find
```

| reading | baseline | after |
|---|--:|--:|
| PANAS inspired | 2.20 | 3.16 |
| PANAS attentive | 2.14 | 2.89 |
| PANAS alert | 1.85 | 2.65 |
| PANAS positive | 2.14 | 2.92 |
| PANAS negative | 2.48 | 2.45 |
| calibrated placement | — | (0.736, 0.378) |

## What the pre-stated criterion picked instead

I said before running that I would select on calibrated placement error. That picks **valley, 8 lines, clean+semantic** at 0.043, which places best but moves the self-report much less (inspired +0.26). Reporting both rather than changing the criterion after seeing the numbers.

```
i thought it might be pan.
in the books you have read.
for nature s sad reality.
can i forget the dear landscape around.
when one is young you know then one can sing.
and the luck will change you say all right.
great guide i ask you still.
a supernatural faith to paint this christ
```

## All candidates

| constructor | lines | filter | garbled | calibrated error | EMA error | Δ inspired | Δ attentive |
|---|--:|---|--:|--:|--:|--:|--:|
| valley | 8 | clean+semantic | 0 | 0.043 | 0.207 | +0.26 | +0.19 |
| polygon-pca | 8 | clean+semantic | 0 | 0.052 | 0.286 | +0.55 | +0.39 |
| polygon-pca | 24 | clean+semantic | 0 | 0.066 | 0.281 | +0.92 | +0.75 |
| valley | 24 | clean+semantic | 0 | 0.094 | 0.248 | +0.96 | +0.75 |
| polygon-pca | 8 | clean | 0 | 0.098 | 0.302 | +0.10 | +0.25 |
| polygon-pca | 8 | unfiltered | 0 | 0.098 | 0.302 | +0.10 | +0.25 |
| valley | 24 | clean | 0 | 0.112 | 0.230 | +0.82 | +0.73 |
| valley | 24 | unfiltered | 0 | 0.112 | 0.230 | +0.82 | +0.73 |
| polygon-pca | 24 | clean | 0 | 0.132 | 0.285 | +0.55 | +0.60 |
| polygon-pca | 24 | unfiltered | 0 | 0.132 | 0.285 | +0.55 | +0.60 |
| valley | 8 | unfiltered | 1 | 0.146 | 0.245 | +0.43 | +0.45 |
| valley | 8 | clean | 0 | 0.154 | 0.229 | +0.44 | +0.41 |
