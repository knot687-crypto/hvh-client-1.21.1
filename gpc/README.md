# Right stick: zero-tunable precision conditioner (Cronus Zen GPC)

`right_stick.gpc` is a self-contained Cronus Zen script for the right stick.
Paste it into Zen Studio, compile it and program a slot. It has no tunable
values: every constant is derived in `tools/derive.py` or is a documented
property of the Zen.

## What it does

The Zen exposes the right stick in two forms:

* `PS4_RX` / `PS4_RY`: X and Y in **whole percent**;
* `get_ipolar(POLAR_RS, …)`: radius on a 0–32767 scale and angle in whole
  degrees.

A script that reads and writes whole percent rounds your stick to 1 % steps.
A thumb held at 12.4 % then turns the camera at 12 % for as long as it is
held. This script:

1. **fuses** the two views into a 0.01 % estimate of the physical stick
   (exact radius, angle constrained by the whole-percent cell);
2. writes that estimate back through the whole-percent channel with a
   **noise-shaping quantiser**: each output is one of the two whole percents
   around the estimate, chosen so that the running sum of output error never
   reaches one step. The camera, which integrates the stick, turns at the
   rate you are actually holding.

What it does **not** do: aim assist, anti-recoil, response curves,
prediction, smoothing or deadzones. A released stick outputs exactly 0. The
left stick and all buttons pass through untouched.

Measured on the simulated Zen (`tools/derive.py`, section 11), RMS
camera-velocity error averaged over 1 s windows:

| | X axis | Y axis |
|---|---|---|
| whole-percent pass-through | 0.0304 % | 0.0366 % |
| this script | 0.0095 % (3.2× lower) | 0.0224 % (1.6× lower) |

These are sub-percent effects. They matter for slow, precise tracking, not
for flicks.

## How it works (one VM cycle, 10 ms by default)

```
get_rtime ──► cross-ratio test ── uniform? ──────────────────────────────┐
PS4_RX/RY ─┐                                                             │
           ├─► fusion: polar arc ∩ integer cell (slab test along the     │
get_ipolar ┘   tangent = voxel ray), 6 rounding hypotheses identified    │
                │                                                        │
                ▼                                                        ▼
        target v (0.01 %) ──► Smith / error-feedback quantiser ──► PS4_RX/RY
                                    ▲
     8x8 blue-noise mask ◄── two golden voxel rays ◄── entropy (stroke restarts)
```

## Where each requested concept is used

*Run* means it executes on the Zen. *Design* means it was used to derive or
verify a constant or a law the script runs, with the numbers reproduced by
`tools/derive.py`.

| Concept | Where | What it does here |
|---|---|---|
| Curse of dimensionality | run + design | The state vector is kept per axis. The joint two-axis Bellman problem has a zero cross term, so the optimal policy factorises. A tabulated DP over the joint state would need about 4.8×10¹⁸ cells. |
| Bellman equations | design | Minimum-variance control of the integrated camera error. The Riccati fixed point gives p = 0 and gain k = 1 (deadbeat), which is the script's `w = v − smith_m`. |
| HJB | design | The same problem in continuous time: gain 1/√μ → ∞ and cost → 0. The one-cycle delay is what leaves the one-sample residual. |
| Control theory | run | Closed error-feedback loop: the integrated error stays below one step (proved invariant), with zero error at DC. |
| Smith predictor | run | Plant = z⁻¹ · quantiser, model = z⁻¹. Feeding back the measured-minus-model mismatch is exactly error feedback (checked on 20 000 samples). |
| Discrete delay | run | The z⁻¹ of that loop; `get_rtime()` timebase. |
| Bode integral theorem | design | NTF S(z) = 1 − z⁻¹ has ∫₀^π ln\|S\| dω = 0: error removed below fs/6 reappears above it, at most ×2 at Nyquist. |
| PSD | design | The measured output-error spectrum rises at 17–19 dB/decade at low frequency (ideal first-order shaping of a white error: 20) and vanishes at DC. The mask and the ray sequences are checked spectrally. |
| Poisson-disk | design | Mask ranks 0–3 are the maximal radius-4 set on the 8×8 torus (4 points, the hexagonal packing bound): (0,0), (4,0), (2,4), (6,4). Ranks 0–7 are ≥ 2 apart and ranks 0–15 ≥ √2 apart. |
| 8×8 mask | run | `RS_MASK`, a 64-level dither matrix. |
| Blue noise, radius 4 | design | The mask is annealed for 2-D blue noise (Georgiev–Fajardo energy) on the radius-4 seed. Its lowest spatial frequencies sit at 0.19–0.23× the power of white noise. |
| Voxel ray | run | (a) Two Amanatides–Woo DDA rays (slope 1/φ, perpendicular) walk the torus-wrapped mask and give each axis its threshold. Below 1/16 of the sample rate their power is 0.055 (X) and 0.078 (Y) of white noise. The X–Y cross-correlation (0.040) is at the level of independent sequences. (b) The fusion is a Kay–Kajiya slab test of the polar arc's tangent ray against the integer cell. |
| Cross ratio | run | (t₀,t₁;t₂,t₃) = 4/3 for evenly spaced samples at any period. The test allows for `get_rtime()`'s 1 ms resolution; if it fails, the integrated error state is discarded. |
| Kolmogorov randomness | run + design | No program can emit an incompressible sequence, so the rays restart every stroke from entropy harvested off the stick. 65 536 thresholds compress to about 0.6–1.1 kB from one deterministic ray, 4.1 kB with stroke restarts (2.8 kB of injected entropy), and 50 kB if i.i.d. |
| Cramér–Rao | design | A still thumb's whole-percent reading has zero Fisher information inside its cell. That is why the polar channel is used instead of estimating over time. Bounded quantisation errors also break the CRB's regularity conditions, which is what lets one sample refute a rounding hypothesis outright. |
| ADMM | design | Solves the ℓ1-constrained noise-shaping QP (optimum 1 − z⁻¹, confirming the Bellman result). It also evaluated and rejected a sliding-window MAP estimator: it lowers per-sample error on a clean ramp (0.292 → 0.165 %) but is worse on realistic hand motion (0.261 → 0.286 %). Not run on the device. |
| Clairaut | design | The objective Hessian is symmetric (mixed partials commute), so one eigendecomposition serves every ADMM penalty, and the two-axis Riccati matrix is symmetric with a zero cross term. |

## Assumptions about the Zen, and what happens if they are wrong

The Zen documentation does not say how the readings are rounded. The script
does not assume it; it **identifies** it:

* the polar angle may be rounded or truncated;
* the whole-percent reading may be rounded, truncated toward zero or floored.

That gives six hypotheses. Every sample refutes any hypothesis it is
inconsistent with (the error budget, `RS_SLACK`, guarantees the true one is
never refuted). In 120 simulated sessions covering all six combinations,
exactly one was left after a median of 33–49 samples of stick movement
(about 0.4 s), 141 at most. The estimate is then the Chebyshev centre of
its feasible set. If all six are refuted, the polar channel is declared
invalid and the script becomes an exact whole-percent pass-through. The
tests cover a mirrored or rotated angle and radius scales 0.7 and 1.3.

Other assumptions: the radius is 32767 at full deflection, the angle
convention is X = r cos a, Y = −r sin a (the `set_polar2` example in the
32-bit update notes), and `get_rtime()` counts whole milliseconds.

## Verification

```
cd gpc/tools
python3 test_right_stick.py     # runs the real .gpc in an interpreter (stdlib only)
python3 derive.py               # every derivation and number above (needs numpy)
python3 derive.py --search      # also re-runs the mask annealing (slow)
```

* `gpcsim.py` interprets the GPC subset the script uses: 32-bit integers,
  truncating division, runtime and const arrays, functions, no locals. It is
  stricter than the device: it rejects any overflow, out-of-range index,
  undeclared name, recursion or local declaration.
* `reference.py` is an independent Python implementation of the
  algorithm. The tests require the interpreted script and the reference to
  agree bit for bit (outputs, error states, entropy pool, hypothesis sets)
  over long simulated sessions.
* `zen_model.py` simulates the Zen, including deliberately wrong
  conventions.

## Limitations

* Not tested on hardware. The script uses only documented GPC constructs
  and functions available on 32-bit firmware, but nothing here can replace
  a compile in Zen Studio and a look at the Device Monitor. The traces show
  the fused targets, the hypothesis count and the error states.
* VM cost: about 1 100 interpreted operations per cycle (worst case about
  1 900, only while the conventions are being identified). On-device VM load
  was not measured.
* The guarantees are about the integrated output, which is what a
  rate-controlled camera sees. A game that samples the stick less often
  than the Zen updates still converges to the right average, but loses
  the one-step bound. The dither is a golden-ratio rotation restarted every
  stroke, so it has no short period for the game's sampling to lock onto.
  In a twin-stick game that reads absolute stick position, the ±1 %
  per-sample dither would be visible as a slight shimmer.
* On Zen Studio Live, `ANALOG_RX/RY` give ±10000 resolution directly. This
  script deliberately uses the polar and whole-percent channels, which are
  available on every 32-bit firmware.
