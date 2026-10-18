# Gesture Lab: Intentions, Requirements and Next Steps

## 1. Intention

Treat the game as an **experiment on a person**, not a difficulty dial.

- The physics tells us exactly which jump times succeed (the window `[L, U]`).
- The player is the unknown: how precisely and how early or late they time a jump.
- The game should present obstacles **on purpose**, in a sequence designed to measure the player, starting from safe assumptions and moving toward their limits.
- The data should let us **check our assumptions** (for example, "is timing error Gaussian?") and build better models, not only report a success rate.

## 2. Where the repo is today

| Goal | Status |
|---|---|
| Varied, non-repetitive experience | Weak. One number `d` drives speed, height, width and spawn interval, with only about 7% jitter. Players tend to sit near `d ≈ 1`, so the game stops changing. |
| Measure the user | Partial. Outcomes (0/1) and gesture latency are logged. **The jump time relative to the obstacle is not logged.** |
| Analysis to build or test models | Partial. Confidence intervals, rolling rates, and a Gaussian σ fit exist. Nothing tests the assumptions (bias, tails, speed effects, history). |

## 3. Terms

- `t = 0`: the instant the obstacle's front edge meets the player's front edge.
- `t_j`: jump (takeoff) time on that clock, in seconds.
- `[L, U]`: window of jump times that clear the obstacle and land before the next one.
- `W = U − L`: window width, the available timing margin.
- `δ = t_j − (L + U)/2`: timing error relative to the window centre.
- Physics in `jump_service.py` and in `dino_jump_success_model.md`.

## 4. Requirements

### 4.1 Data (must have)

For every obstacle, log:

| Field | Meaning |
|---|---|
| `t_j` | jump time relative to the obstacle clock; empty if the player never jumped |
| `L`, `U`, `W` | window from physics, using the actual height, width, speed and next gap |
| `binding` | which bound limits `U`: clearance or gap |
| `delta` | `t_j − (L+U)/2` |
| `outcome` | success or failure |
| `failure_mode` | `early` (`t_j < L`), `late` (`t_j > U`), or `no_jump` |
| `next_gap_px` | gap to the following obstacle |

Rule for assigning a jump to an obstacle: the accepted jump whose takeoff is the last one before the obstacle's overlap with the player starts.

### 4.2 Obstacle sequence (must have)

- A **fixed, designed list** of obstacles, not driven by `d`.
- Speed stays constant in the first version. Only height varies.
- Heights are chosen to produce a spread of window widths, from wide to narrow. For a target `W`, with `τ = (w_o + w_p)/s`:
  ```
  r = g (W + τ) / 2          h = (v₀² − r²) / (2g)
  ```
- Order is shuffled, with easy obstacles interleaved, so the experience is not repetitive.
- Gaps are large enough that the clearance bound is the binding one.
- Every obstacle must be feasible (`W > 0`). A failure should never come from an impossible obstacle.

### 4.3 Analysis (must have)

1. Success rate against `W`, with Wilson confidence intervals.
2. Fit of the timing bias `μ` and spread `σ` from the logged `δ` values (not only the 0/1 outcomes).
3. QQ plot of `δ` to check whether the Gaussian is reasonable.
4. Failure mode counts (early, late, no jump).

### 4.4 Constraints

- `adaptive.py` and `gesture_lab` logic stay free of pygame, cv2 and mediapipe imports.
- Respect `AGENTS.md` invariants. Adding columns to `OUTCOME_HEADER` is allowed only by **appending** them at the end, with the tests updated together.
- The existing controller and `d` stay in the code and are simply unused in experiment mode.
- Safety: every obstacle is feasible, easy obstacles are mixed in, and no run ends on a long streak of hard ones.

## 5. Out of scope for now

Speed variation, gap (recovery) probes, reaction-time probes, adaptive staircases, new obstacle types, multi-player comparison, and any change to the controller. Each of these is the same pattern with a different knob and can be added later as a new obstacle list.

## 6. Simplified next steps

1. **Log the jump time.** Add the fields in 4.1. Touches logging only.
2. **Add an experiment mode.** A fixed, shuffled list of about 30 obstacles with varied heights (4.2). Selected with a command-line option.
3. **Add one analysis script.** Success against `W`, fit of `μ` and `σ`, QQ plot, failure modes (4.3).
4. **Run it on a few players** and look at the results before deciding on anything else.

Only after step 4: choose the next limit to measure (gap, speed, then reaction time), one at a time.

## 7. Open questions

- How many obstacles can one session contain (about 30 assumed)?
- Is appending columns to `OUTCOME_HEADER` acceptable, or should the trial data go to a separate log file?
- Which window widths are safe as the easiest and hardest starting points for your players? The first run can answer this.
