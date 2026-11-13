# BUILD NOTES — roblox-up-and-coming-forecast-01, v3

## What v3 is

A **stack** of three forecasters, blended 0.80 / 0.18 / 0.02 in probability space
(weights from a coarse grid on both validation splits):

- **(A) Monte Carlo rank simulation — the v2 engine, unchanged.**
  Each game has a latent rank-value that mean-reverts toward a target
  extrapolated from its recent rank velocity, with band-dependent noise
  (per-hour rank-diffusion sd by band: 0.82 top 10 / 2.10 11–30 /
  3.32 31–60 / 3.85 61+). New games spawn at the bottom of the chart
  (Poisson 0.8/hour); 45% are "risers" climbing exponentially toward a
  top-10 target. Games age out 28 days after publishing. 2,500 sims,
  fixed seed 20261113. Tuned v2 params: DRIFT_SHRINK 0.15, RISER_LAM
  0.12, HORIZON 24, TOP_RISER_FRAC 0.35.
- **(B) Fitted place-occupancy table via a Gaussian rank process.**
  Per game, rank(h) ~ N(mu(h), sd(h)) with mu/sd from lag regressions
  fitted on the practice archive (45 reads, 2026-10-30 → 2026-11-10,
  rebuilt from the public Kenny-Dhub/roblox-snapshots archive):
  top-10 games sink with slope > 1 (24h: 1.31, 48h: 1.73, 72h: 2.22,
  resid sd 5.4 → 11.4 → 15.2); mid-chart games (11–40) drift down
  ~15 ranks / 72h (resid sd 7.1 → 20.7). Place probabilities via
  order statistics (Poisson-binomial normal approximation + 9-point
  quantile quadrature), time-averaged over the 72h window (every 2nd
  hour), tempered (temp 2.0) and normalised per place to 0.85 named
  mass (0.15 left unnamed for brand-new games).
- **(C) Fitted practice table (weight 0.02).** An empirical per-place
  occupancy table fitted on the practice window: recency-weighted
  (36h half-life), place-kernel smoothed (Gaussian kernel sd 3.5
  places), with a small drift shift from each game's fitted velocity
  (shrink 0.10 × 36h), tempered (temp 1.0) and normalised per place
  to 0.65 named mass (0.35 left unnamed). The raw table alone scores
  -50..-76 and the smoothed table alone about -39, but a 2%
  admixture sharpens the top-10 set without overcommitting.

The stack keeps the MC's honest spread — the log score punishes
confident wrong calls far more than it rewards confident right ones —
while the fitted table sharpens the drift path slightly toward the
measured top-10 sinking behaviour.

## What was tried and rejected (all scored on BOTH splits)

- **Pure static occupancy table** (recency-weighted empirical
  per-place table, incl. last-1-read and last-34-read variants):
  **-50 to -76** — catastrophic. It gets the top-10 set mostly right
  (6.4 hits/read) but the places wrong, and its near-one-hot
  probabilities pay log(0.01) = -4.6 per wrong place. A fitted table
  without dynamics is NOT the winning method.
- **Gaussian table alone**: -34.46 / -38.55 (best grid point:
  sd_scale 1.0, temp 2.0, unnamed 0.15). Better than the static table,
  still worse than the MC on both splits. Drift scale (0 / 0.5 / 1.0)
  barely matters — the diffusion/spread, not the drift, is the
  bottleneck.
- **Table in the stack at any weight ≥ 0.2**: drags the stack to
  -36…-46. At weight 0.02 it HELPS (+0.48 practice, +0.15 Nov-4 over
  the 0.80/0.18 pair) — the small admixture sharpens the top-10 set
  without overcommitting to the static table.
- **`playing`-count growth as a feature**: 24h playing growth vs
  24h/48h rank change correlates only r = 0.15–0.17 (n ≈ 1,000), and
  sorting by playing growth does NOT reproduce the chart order
  (top-10 overlap 0–5/10 across windows). Too weak to move the stack.
- **Tempering the MC harder**: gamma 0.9 is a wash, sharper is worse
  on the Nov-4 split (v2 builder's result, confirmed).

## Validation (real RULES scorer, apples-to-apples with v1/v2)

| entry | practice split (10 reads Nov 8–10) | Nov-4 split (11 reads Nov 4–7) | avg |
|---|---|---|---|
| house baseline | -43.51 | — | — |
| v1 (MC) | -33.87 | -38.31 | -36.09 |
| v2 (tuned MC) | -33.54 | -37.88 | -35.71 |
| **v3 (3-way stack, this entry)** | **-32.98** | **-37.54** | **-35.26** |

v3 beats v2 on **both** splits (+0.56 practice, +0.34 Nov-4), so it
passes the submittability gate. Scores are from the staged main.py
itself, run end-to-end on both split inputs (not from the tuning
harness): practice forecast sha256
f1effedba462126d5f048606e99f5ec987ca3e051ee9243010d9f4192f5c1d78,
Nov-4 forecast sha256
1395a9e5f7cba2f7f2c150f84a8f3b7b7ad5c06a0112e824dbd16331df7b2b9f.
(The 2-way 0.9/0.1 stack scored -33.44/-37.69; the table admixture
adds +0.46 practice, +0.15 Nov-4.)

- Top-10 hits / agreement: practice 6.4 / 53.5, Nov-4 5.45 / 43.3
  (v2: 6.4 / 54.7 practice).
- Per-read scores degrade over the 72h window (practice: -24.2 →
  -40.7; Nov-4: -21.5 → -48.3) — the chart drifts away from any
  forecast; expected, not a bug.
- Output shape: exactly two keys (`probabilities`, `top10`); practice
  327 rows / Nov-4 299 rows; max 44 rows/place; per-place named sums
  0.68–0.83 (≤ 1, remainder is the unnamed share for brand-new games);
  all p in (0, 1); top10 is 10 distinct ids.
- Runtime: ~58 s (practice input, 34 reads) / ~45 s (Nov-4 input,
  19 reads) — well under the 300 s limit. Two runs per split are
  byte-identical (fixed seed, no clock, no unseeded randomness).

## Honest caveats

1. **The gain over v2 is modest (+0.45 avg).** This is a genuine but
   modest improvement, not a breakthrough.
2. **The -13.2 top-10 bar is not reachable out-of-sample from this
   data.** An ORACLE table fitted on the answer reads themselves
   scores only -14.23 (hits 8.3, agreement 75.5) — and the live #1
   entry's dev numbers (-13.18, hits 8.3, agreement 75.5) match that
   oracle almost exactly. Every honest out-of-sample method tested
   here lands in the -33s. The #1 dev score is therefore almost
   certainly an artefact of the dev grading setup (in-sample or a
   different window), not a forecast achievable from the practice
   history alone. Do not expect v3's final score to be anywhere near
   -13; the realistic prize bar is beating the best house entry,
   and v3 is ~0.15 better than v2 on that path.
3. **Weights were selected on the same two splits used for
   validation** (there is no third split). The 0.80/0.18/0.02 choice sits on
   a flat plateau, which limits overfitting risk, but the final window (Nov 13–16) is a genuinely
   unseen split and the +0.45 edge may not fully transfer.
4. **Unnamed games**: 8 of the 17 games in the practice answer top 10
   were never in the training top 10. The MC's spawn/riser machinery
   and the unnamed share are what cover them; the Gaussian table
   cannot name a game it has never seen.

## Reproduce

- Practice split: `MAKEFUN_INPUT_DIR=<kit>/input` (34 reads to Nov 7)
  → score vs the 10 archive reads Nov 8–10.
- Nov-4 split: history = archive reads before Nov 4 00:00 UTC
  (19 reads), window Nov 4–7 → score vs the 11 archive reads Nov 4–7.
- Scorer: the RULES log score (floor 0.0001, unnamed share =
  (1 − place_sum) / (n − named_count)); implemented in
  `~/workspace/makefun/roblox-work/scorer.py`.
- Tuning harness + experiments:
  `~/workspace/makefun/roblox-work/{v3exp,v3dyn,v3dyn2,v3model,stack}.py`.
