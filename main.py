#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming top 10 for make.fun bounty
roblox-up-and-coming-forecast-01. v3 (see BUILD_NOTES.md).

Method: a STACK of three independently fitted forecasters, blended
0.80 / 0.18 / 0.02 in probability space (weights chosen on the practice
split and the Nov-4 split):

  (A) Monte Carlo rank simulation (the v2 engine, unchanged):
      each game has a latent rank-value that mean-reverts toward a
      target extrapolated from its recent rank velocity, with
      band-dependent noise. New games spawn at the bottom of the
      chart; a fraction are "risers" that climb exponentially toward
      a top-10 target. 2500 sims, fixed seed.

  (B) Fitted place-occupancy table via a Gaussian rank process:
      per game, rank(h) ~ N(mu(h), sd(h)) with mu/sd from lag
      regressions fitted on the practice archive (top-10 games sink
      with slope > 1; mid-chart games drift down ~15 ranks / 72h;
      residual sd grows with horizon). Place probabilities come from
      order statistics (Poisson-binomial normal approximation +
      quantile quadrature), time-averaged over the 72h window, then
      tempered (temp 2.0) and normalised per place to 0.85 named
      mass (0.15 left unnamed for brand-new games).

The stack keeps the MC's honest spread (the log score punishes
confident wrong calls far more than it rewards confident right ones)
while the fitted Gaussian table sharpens the drift path slightly
toward the measured top-10 sinking behaviour, and the empirical
practice table (recency-weighted, place-kernel smoothed, small drift
shift; weight 0.02) sharpens the top-10 set further. A pure static
occupancy table was tested and REJECTED: it gets the top-10 set
mostly right but the places wrong, and its near-one-hot probabilities
score -50..-76. The Gaussian table alone scores -34.5/-38.6; the
3-way stack scores -32.98/-37.54, beating the 2-way stack
(-33.44/-37.69) and v2 (-33.54/-37.88) on both splits.

Stdlib only. Reads $MAKEFUN_INPUT_DIR (default /input), writes
$MAKEFUN_OUTPUT_DIR (default /output)/<task.output>.
"""
import json
import math
import os
import random
from datetime import datetime, timedelta

# ---- component A: Monte Carlo (v2 parameters, unchanged) ----
N_SIMS = 2500
SEED = 20261113
SIGMA_SCALE = 1.0       # scales band sigmas
DRIFT_SHRINK = 0.15     # regression slope of rank velocity on its next value
LAM = 0.05              # mean-reversion rate for ordinary games (/hour)
RISER_LAM = 0.12        # mean-reversion rate for risers (/hour)
RISER_FRAC = 0.45       # fraction of new spawns that are risers
TOP_RISER_FRAC = 0.35   # fraction of risers targeting places 1-2
SPAWN_RATE = 0.8        # brand-new games entering the chart per hour
HORIZON = 24.0          # hours to extrapolate rank velocity for targets
MAX_ROWS_PER_PLACE = 100
# per-hour rank-diffusion sd by rank band (measured on the archive)
SIGMA = {1: 0.82, 11: 2.10, 31: 3.32, 61: 3.85}

# ---- component B: fitted Gaussian table ----
GAUSS_SD_SCALE = 1.0
GAUSS_TEMP = 2.0
GAUSS_UNNAMED = 0.15
GAUSS_VEL_SHRINK = 0.1
GAUSS_HOUR_STEP = 2     # evaluate every 2nd hour, time-averaged
# ---- component C: fitted practice table ----
TABLE_HALF_LIFE_H = 36.0
TABLE_SIGMA_PLACE = 3.5
TABLE_TEMP = 1.0
TABLE_UNNAMED = 0.35
TABLE_DRIFT_SHRINK = 0.10

# ---- stack weights (fixed; see header note on self-calibration) ----
STACK_W_MC = 0.80
STACK_W_GAUSS = 0.18
STACK_W_TABLE = 0.02

# fitted lag regressions (measured on the practice archive):
# rank<=10: E[r(t+h)] = s(h)*r0 + b(h), resid sd rs(h)
TOP_S = [(0, 1.0, 0.0, 0.5), (24, 1.307, 0.95, 5.40),
         (48, 1.731, 3.95, 11.37), (72, 2.218, 6.86, 15.15)]
# rank 11..40: E[r(t+h)] = r0 + d(h), resid sd rs(h)
MID_D = [(0, 0.0, 0.5), (12, 1.11, 7.08), (24, 2.88, 12.16),
         (48, 8.38, 17.69), (72, 15.09, 20.65)]


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def sigma_for(rank):
    for lo in (61, 31, 11, 1):
        if rank >= lo:
            return SIGMA[lo]
    return SIGMA[1]


def load_history(input_dir):
    snaps = []
    with open(os.path.join(input_dir, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if line:
                snaps.append(json.loads(line))
    snaps.sort(key=lambda s: s["at"])
    task = json.load(open(os.path.join(input_dir, "task.json")))
    return snaps, task


def build_state(snaps):
    """Per-game state from the history."""
    times = [parse_time(s["at"]) for s in snaps]
    series = {}
    first_seen = {}
    created = {}
    for i, s in enumerate(snaps):
        for g in s.get("up_and_coming") or []:
            series.setdefault(g["universe_id"], []).append(
                (times[i], g["rank"]))
            first_seen.setdefault(g["universe_id"], times[i])
        det = s.get("details") or {}
        for k, v in det.items():
            if v and v.get("created"):
                created[int(k)] = parse_time(v["created"])
    last = {g["universe_id"] for g in snaps[-1].get("up_and_coming") or []}
    games = []
    for uid, obs in series.items():
        obs.sort()
        r_last = obs[-1][1]
        v = 0.0
        if len(obs) >= 2:
            (t1, r1), (t2, r2) = obs[-2], obs[-1]
            dt = (t2 - t1).total_seconds() / 3600.0
            if dt > 0:
                v = (r2 - r1) / dt
        ce = created.get(uid)
        if ce is None:
            ce = first_seen[uid] - timedelta(days=3)
        games.append({
            "uid": uid,
            "x": float(r_last),
            "v": v,
            "exit": ce + timedelta(days=28),
        })
    return games, times[-1], len(last)


def simulate(games, t0, n_chart, hours, rng):
    counts = {}
    anon_counts = [0] * 11
    total_slots = N_SIMS * hours

    def entry_rank():
        u = rng.random()
        if u < 0.5:
            return rng.uniform(80, 97)
        if u < 0.9:
            return rng.uniform(60, 80)
        return rng.uniform(35, 60)

    def riser_target():
        if rng.random() < TOP_RISER_FRAC:
            return rng.uniform(1, 2)
        return rng.uniform(3, 10)

    for _ in range(N_SIMS):
        pool = []
        for g in games:
            v = max(-1.5, min(1.5, g["v"] * DRIFT_SHRINK))
            if v < -0.5:
                target = max(1.0, g["x"] + v * 24.0)
                lam = RISER_LAM
            else:
                target = max(1.0, min(115.0, g["x"] + v * HORIZON))
                lam = LAM
            pool.append([g["x"], target, lam, g["uid"], g["exit"]])
        for h in range(hours):
            now = t0 + timedelta(hours=h + 1)
            # Poisson(spawn_rate) new games
            lam_p = math.exp(-SPAWN_RATE)
            k = 0
            p = 1.0
            while True:
                p *= rng.random()
                if p <= lam_p:
                    break
                k += 1
            for _ in range(k):
                er = entry_rank()
                if rng.random() < RISER_FRAC:
                    pool.append([er, riser_target(), RISER_LAM, None, None])
                else:
                    pool.append([er, rng.uniform(80, 100), LAM, None, None])
            alive = []
            for e in pool:
                if e[4] is not None and now >= e[4]:
                    continue
                sig = sigma_for(max(1, int(e[0]))) * SIGMA_SCALE
                e[0] += e[2] * (e[1] - e[0]) + rng.gauss(0, sig)
                alive.append(e)
            pool = alive
            pool.sort(key=lambda e: e[0])
            for place in range(1, 11):
                if place - 1 < len(pool):
                    uid = pool[place - 1][3]
                    if uid is None:
                        anon_counts[place] += 1
                    else:
                        c = counts.setdefault(uid, [0] * 11)
                        c[place] += 1
    probs = {}
    for uid, c in counts.items():
        for place in range(1, 11):
            p = c[place] / total_slots
            if p > 0:
                probs.setdefault(place, []).append((uid, p))
    unnamed = {place: anon_counts[place] / total_slots
               for place in range(1, 11)}
    return probs, unnamed


def build_forecast(probs, last_chart_uids):
    rows = []
    for place in range(1, 11):
        top = sorted(probs.get(place, []), key=lambda t: (-t[1], t[0]))
        for uid, p in top[:MAX_ROWS_PER_PLACE]:
            rows.append({"universe_id": uid, "place": place,
                         "p": round(p, 6)})
    # top10: greedy assignment maximising summed probability
    cand = []
    for place in range(1, 11):
        for uid, p in probs.get(place, []):
            cand.append((p, uid, place))
    cand.sort(key=lambda t: (-t[0], t[1], t[2]))
    used_g, used_p, top = set(), set(), {}
    for p, uid, place in cand:
        if uid in used_g or place in used_p:
            continue
        used_g.add(uid)
        used_p.add(place)
        top[place] = uid
        if len(top) == 10:
            break
    top10 = [top[k] for k in sorted(top)]
    if len(top10) < 10:  # fallback: fill from the last chart order
        for uid in last_chart_uids:
            if uid not in top10:
                top10.append(uid)
            if len(top10) == 10:
                break
    return {"probabilities": rows, "top10": top10[:10]}


# ---- component B helpers ----

def _interp(pts, h, idx):
    if h <= pts[0][0]:
        return pts[0][idx]
    for i in range(len(pts) - 1):
        h0, h1 = pts[i][0], pts[i + 1][0]
        if h0 <= h <= h1:
            f = (h - h0) / (h1 - h0)
            return pts[i][idx] + f * (pts[i + 1][idx] - pts[i][idx])
    return pts[-1][idx]


def _gauss_mu_sd(r0, h, vel):
    if r0 <= 10:
        s = _interp(TOP_S, h, 1)
        b = _interp(TOP_S, h, 2)
        sd = _interp(TOP_S, h, 3)
        mu = s * r0 + b
    else:
        d = _interp(MID_D, h, 1)
        sd = _interp(MID_D, h, 2)
        mu = r0 + d
    mu += GAUSS_VEL_SHRINK * vel * min(h, 24)
    mu = max(1.0, mu)
    sd = max(0.5, sd)
    return mu, sd


def _ndtri(p):
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343744e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


_SQRT2 = math.sqrt(2.0)


def _phi_cdf(x):
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


_QNODES = [(i + 0.5) / 9 for i in range(9)]
_QZ = [_ndtri(q) for q in _QNODES]


def _gauss_place_probs_at_hour(games, t0, h):
    """Dict (uid, place) -> p for this hour via order statistics."""
    alive = []
    for g in games:
        if g["exit"] is not None and t0 + timedelta(hours=h) >= g["exit"]:
            continue
        mu, sd = _gauss_mu_sd(g["x"], h, g["v"])
        alive.append((g["uid"], mu, sd * GAUSS_SD_SCALE))
    if not alive:
        return {}
    acc = {}
    for uid, mu, sd in alive:
        for z in _QZ:
            x = mu + sd * z
            m = 0.0
            vv = 0.0
            for _u2, mu2, sd2 in alive:
                q = _phi_cdf((x - mu2) / sd2)
                m += q
                vv += q * (1.0 - q)
            qself = _phi_cdf(z)
            m -= qself
            vv -= qself * (1.0 - qself)
            s = math.sqrt(max(vv, 1e-9))
            w = 1.0 / len(_QZ)
            for k in range(1, 11):
                pk = _norm_pdf((k - 1 - m) / s) / s
                acc[(uid, k)] = acc.get((uid, k), 0.0) + w * pk
    return acc


def gauss_forecast(games, t0, hours, last_chart_uids):
    """Fitted Gaussian place-occupancy table, time-averaged."""
    total = {}
    hs = list(range(1, hours + 1, GAUSS_HOUR_STEP))
    for h in hs:
        acc = _gauss_place_probs_at_hour(games, t0, h)
        for key, p in acc.items():
            total[key] = total.get(key, 0.0) + p / len(hs)
    rows = []
    for place in range(1, 11):
        items = [(uid, p) for (uid, pl), p in total.items() if pl == place]
        if not items:
            continue
        items = [(uid, p ** GAUSS_TEMP) for uid, p in items]
        s = sum(p for _, p in items)
        items.sort(key=lambda t: -t[1])
        for uid, p in items[:MAX_ROWS_PER_PLACE]:
            pp = p / s * (1.0 - GAUSS_UNNAMED)
            if pp >= 0.0005:
                rows.append({"universe_id": uid, "place": place,
                             "p": round(pp, 6)})
    cand = sorted(((r["p"], r["universe_id"], r["place"]) for r in rows),
                  reverse=True)
    used_g, used_p, top = set(), set(), {}
    for p, uid, place in cand:
        if uid in used_g or place in used_p:
            continue
        used_g.add(uid)
        used_p.add(place)
        top[place] = uid
        if len(top) == 10:
            break
    top10 = [top[k] for k in sorted(top)]
    for uid in last_chart_uids:
        if len(top10) >= 10:
            break
        if uid not in top10:
            top10.append(uid)
    return {"probabilities": rows, "top10": top10[:10]}


# ---- component C: fitted practice table ----

def table_forecast(snaps, games, last_chart_uids):
    """Empirical per-place occupancy table with place-kernel smoothing
    and a small drift shift. Returns a forecast dict like the others."""
    t_end = parse_time(snaps[-1]["at"])
    vel = {g["uid"]: g["v"] for g in games}
    counts = {k: {} for k in range(1, 11)}
    koff = range(-6, 7)
    for r in snaps:
        age = (t_end - parse_time(r["at"])).total_seconds() / 3600.0
        w = 0.5 ** (age / TABLE_HALF_LIFE_H) if TABLE_HALF_LIFE_H else 1.0
        for g in r.get("up_and_coming") or []:
            if g["rank"] > 10:
                continue
            uid = g["universe_id"]
            centre = g["rank"] + TABLE_DRIFT_SHRINK * vel.get(uid, 0.0) * 36.0
            for off in koff:
                pl = g["rank"] + off
                if pl < 1 or pl > 10:
                    continue
                kw = math.exp(-0.5 * ((pl - centre) / TABLE_SIGMA_PLACE) ** 2)
                if kw <= 0.01:
                    continue
                d = counts[pl]
                d[uid] = d.get(uid, 0.0) + w * kw
    rows = []
    for place in range(1, 11):
        d = counts[place]
        tot = sum(d.values())
        if tot <= 0:
            continue
        items = [(uid, (c / tot) ** TABLE_TEMP) for uid, c in d.items()]
        s = sum(p for _, p in items)
        items.sort(key=lambda t: -t[1])
        for uid, p in items[:MAX_ROWS_PER_PLACE]:
            pp = p / s * (1.0 - TABLE_UNNAMED)
            if pp >= 0.0005:
                rows.append({"universe_id": uid, "place": place,
                             "p": round(pp, 6)})
    cand = sorted(((r["p"], r["universe_id"], r["place"]) for r in rows),
                  reverse=True)
    used_g, used_p, top = set(), set(), {}
    for p, uid, place in cand:
        if uid in used_g or place in used_p:
            continue
        used_g.add(uid)
        used_p.add(place)
        top[place] = uid
        if len(top) == 10:
            break
    top10 = [top[k] for k in sorted(top)]
    for uid in last_chart_uids:
        if len(top10) >= 10:
            break
        if uid not in top10:
            top10.append(uid)
    return {"probabilities": rows, "top10": top10[:10]}


def stack_forecasts(mc_fc, gauss_fc, table_fc, last_chart_uids):
    """Convex blend of the three component forecasts in probability space."""
    def to_map(fc):
        return {(r["universe_id"], r["place"]): r["p"]
                for r in fc["probabilities"]}

    mm = to_map(mc_fc)
    gm = to_map(gauss_fc)
    tm = to_map(table_fc)
    comb = {}
    for k in set(mm) | set(gm) | set(tm):
        comb[k] = (STACK_W_MC * mm.get(k, 0.0)
                   + STACK_W_GAUSS * gm.get(k, 0.0)
                   + STACK_W_TABLE * tm.get(k, 0.0))
    rows = []
    for place in range(1, 11):
        items = [(uid, p) for (uid, pl), p in comb.items() if pl == place]
        items.sort(key=lambda t: (-t[1], t[0]))
        for uid, p in items[:MAX_ROWS_PER_PLACE]:
            if p > 0:
                rows.append({"universe_id": uid, "place": place,
                             "p": round(p, 6)})
    cand = [(p, uid, place) for (uid, place), p in comb.items()]
    cand.sort(key=lambda t: (-t[0], t[1], t[2]))
    used_g, used_p, top = set(), set(), {}
    for p, uid, place in cand:
        if uid in used_g or place in used_p:
            continue
        used_g.add(uid)
        used_p.add(place)
        top[place] = uid
        if len(top) == 10:
            break
    top10 = [top[k] for k in sorted(top)]
    if len(top10) < 10:
        for uid in last_chart_uids:
            if uid not in top10:
                top10.append(uid)
            if len(top10) == 10:
                break
    return {"probabilities": rows, "top10": top10[:10]}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    snaps, task = load_history(input_dir)
    window = task.get("window", {})
    hours = int(window.get("slots", 72))
    games, t0, n_chart = build_state(snaps)
    last_uids = [g["universe_id"]
                 for g in sorted(snaps[-1].get("up_and_coming") or [],
                                 key=lambda r: r["rank"])]
    rng = random.Random(SEED)
    probs, _unnamed = simulate(games, t0, n_chart, hours, rng)
    mc_fc = build_forecast(probs, last_uids)
    gauss_fc = gauss_forecast(games, t0, hours, last_uids)
    table_fc = table_forecast(snaps, games, last_uids)
    forecast = stack_forecasts(mc_fc, gauss_fc, table_fc, last_uids)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, task.get("output", "forecast.json"))
    with open(out_path, "w") as f:
        f.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()
