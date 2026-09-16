# The two showcase poems

Both aim at `focused` = (0.65, 0.60), 24 lines, seed 42, neutral start. **Identical for every model** — the constructors read only the phrase graph and the NRC lexicon, never a model. What changes across models is where the poem lands them.

## valley

poem NRC mean (0.67, 0.39) · body arousal 0.394 → closing four lines 0.600 · 24/24 distinct lines

> yea and in quiet sleep
> quiet as a moonbeam
> i pine for rest
> her eyes blue heavens were serene with soul
> wherein i dwell serene
> a time of peaceful prayer
> the quiet countryside
> autumn leaves autumn leaves
> the soft reiterations sweep
> i would cut a piece from the evening sky
> his dame and his two beauteous little children
> did all the useful package hold
> he can t recall the creature s name
> with wild spring meanings hill and plain together
> such to perfection one first matter all
> exalt thy tow ry head and lift thy eyes
> his good mama was angry quite
> thrilling the world with lightning s vivid wand
> and that the giant wave democracy
> want want want want it hung round everywhere
> what banquet but revenge can glad my mind
> the army of the stars appear
> a warrior s god in glory s clarion calls
> and jack did quickly follow

| model | published metric | calibrated (argmax layer) | calibrated (deep layer) |
|---|--:|--:|--:|
| llama1b | 0.219 | 0.196 | 0.215 |
| gemma2b | 0.127 | 0.159 | 0.170 |
| gemma9b | 0.111 | 0.129 | 0.210 |

## polygon-pca

poem NRC mean (0.57, 0.55) · body arousal 0.550 → closing four lines 0.595 · 24/24 distinct lines

> the courier aquiline so swiftly gone
> atrides then his silver studded sword
> one of that saintly murderous brood
> if inference and reason shun
> tenfold increased he ll reap who has foregone
> be it of war or peace or hate or love
> which i as freely give hell shall unfould
> to heat the soldering irons
> the shepherd s slender strain
> sudden as sweet
> god knows what end the strife will take
> get busy massa willie
> where beauty walks with naked face
> where wild flowers welcome the wandering bee
> over particular remember this caution of martial
> i ll kneel in loving reverent awe
> admire and hate thy blooming years
> with a rocket s sullen glow
> because in the great future buried deep
> but i shall hear thy wild triumphant voice
> the fatal issue to his health fame peace
> a happier home to him is fate cruel
> our fainting hopes in vain revive
> and that the giant wave democracy

| model | published metric | calibrated (argmax layer) | calibrated (deep layer) |
|---|--:|--:|--:|
| llama1b | 0.293 | 0.066 | 0.060 |
| gemma2b | 0.193 | 0.133 | 0.076 |
| gemma9b | 0.214 | 0.034 | 0.087 |
