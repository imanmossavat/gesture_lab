# Predicting Jump Success in a Side-Scrolling Runner: From First Principles

## 1. Idea in one paragraph

Whether a jump succeeds is almost deterministic once you know *when* the player jumps: it is a window check against physics. The only random ingredient is the player's jump time. So we split the problem into two parts:

1. **Physics** gives a hard window `[L, U]` of successful jump times.
2. **A player model** gives the distribution of the jump time `t_j`, written as a CDF `F`.

Then

```
P(success | x, h) = F(U | x, h) − F(L | x, h)
```

The methods below differ only in how they model `F`.

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| `v₀` | vertical launch speed of the jump |
| `g` | gravity |
| `s` | horizontal game speed (scroll speed) |
| `w_p`, `w_o` | player and obstacle **effective collision** widths (the extent that actually counts for contact) |
| `h_o` | obstacle height |
| `d` | gap from the back of this obstacle to the front of the next one |
| `t_j` | jump time (takeoff instant) |
| `t = 0` | the instant the obstacle's front edge reaches the dino's front edge if the dino stayed on the ground |
| `x` | state observed before the jump (`s`, obstacle sizes, `d`, time since landing, lead time, ...) |
| `h` | history (e.g. the previous `t_j`, previous error, previous outcome) |

### Derived quantities

| Symbol | Definition | Meaning |
|---|---|---|
| `τ` | `(w_o + w_p)/s` | time the dino overlaps the obstacle horizontally (first contact to last contact) |
| `r` | `√(v₀² − 2g·h_o)` | defined only if `v₀² ≥ 2g·h_o` |
| `t₁` | `(v₀ − r)/g` | time after takeoff when the dino rises above `h_o` |
| `t₂` | `(v₀ + r)/g` | time after takeoff when it drops back below `h_o` |
| `T_air` | `2v₀/g = t₁ + t₂` | total airtime |
| `T_above` | `t₂ − t₁ = 2r/g` | time spent above `h_o` |

### Derivation of `r` and its physical meaning

Only two physical laws are used: constant gravitational acceleration, and constant horizontal velocity (so horizontal and vertical motion decouple).

**Vertical motion.** After takeoff at `t = 0` with upward speed `v₀`:

```
y(t) = v₀t − ½gt²,      v(t) = v₀ − gt
```

Eliminating `t` (energy conservation per unit mass: `½v² + gy = ½v₀²`):

```
v² = v₀² − 2g·y
```

**Apex.** Set `v = 0`: `H = v₀²/(2g)`, reached at `t = v₀/g`.

**Crossing the obstacle top.** Set `y = h_o` in `y(t)`:

```
½gt² − v₀t + h_o = 0   ⇒   t = (v₀ ± √(v₀² − 2g·h_o)) / g
```

The square root is what we call `r`:

```
r = √(v₀² − 2g·h_o)
```

**Meaning of `r`.** From `v² = v₀² − 2g·y` at `y = h_o`, `r` is the **magnitude of the vertical velocity when the dino is at the obstacle's top height**. It is `+r` on the way up and `−r` on the way down (same speed, opposite direction, by symmetry of constant gravity). Equivalently,

```
r² = 2g(H − h_o)
```

so `r` measures how far the obstacle top sits **below the apex**: the more headroom, the larger `r`.

**Consequences.**

- `t₁ = (v₀ − r)/g` is the time for the upward speed to drop from `v₀` to `r`.
- `t₂ = (v₀ + r)/g` is the time for it to drop from `v₀` to `−r`.
- `T_above = t₂ − t₁ = 2r/g = 2√(2(H − h_o)/g)` is the time to decelerate from `+r` to `−r`. It is the time spent above `h_o`: long if the obstacle is low, shrinking to zero as `h_o → H`.
- `t₁ + t₂ = 2v₀/g = T_air`, and `t₁` is also the time spent below `h_o` on the way down (symmetry).
- Limits: `h_o = 0` gives `r = v₀` and `T_above = T_air`. `h_o = H` gives `r = 0`, a single instant at the apex. `h_o > H` makes `r` imaginary: the obstacle top is never reached.

**Clearability.** Clearing requires the time above `h_o` to cover the overlap, `2r/g ≥ τ`, i.e. `r ≥ gτ/2`. In words: at the obstacle's top height the dino must still be moving vertically at least as fast as the speed gained by free fall over half the overlap time. Squaring, `v₀² − 2g·h_o ≥ g²τ²/4`, gives

```
h_o ≤ v₀²/(2g) − g·τ²/8,   τ = (w_o + w_p)/s
```

This is stricter than `h_o ≤ H` (apex condition only) because the dino must also stay up long enough. It depends on `s` through `τ`: a faster horizontal speed shortens `τ`, so taller obstacles become clearable. As `s → ∞` it reduces to `h_o ≤ H`.

---

## 3. Hard constraints

### 3.1 Clearability

From the derivation in Section 2, the obstacle is clearable iff `T_above ≥ τ`, i.e. `h_o ≤ v₀²/(2g) − g·τ²/8` with `τ = (w_o + w_p)/s`.

### 3.2 The success window `[L, U]`

The obstacle occupies the dino's horizontal position during `t ∈ [0, τ]`. With takeoff at `t_j`, the dino is above `h_o` during `[t_j + t₁, t_j + t₂]`.

- **Not too early** (still above `h_o` when the overlap ends): `t_j + t₂ ≥ τ`

  ```
  L = τ − t₂
  ```

- **Not too late, clearance** (already above `h_o` when the overlap starts): `t_j + t₁ ≤ 0`, so `t_j ≤ −t₁`.

- **Not too late, gap** (land before the next obstacle's front arrives at time `(w_o + d)/s`): `t_j + T_air ≤ (w_o + d)/s`.

Therefore

```
L = τ − t₂
U = min( −t₁ ,  (w_o + d)/s − T_air )
success  ⇔  L ≤ t_j ≤ U
```

The window is nonempty iff `L < U`.

### 3.3 Window width and which constraint binds

| Regime | Width `W = U − L` | Dependence on `s` |
|---|---|---|
| Clearance-limited (`U = −t₁`) | `2r/g − (w_o + w_p)/s` | **grows** with `s` |
| Gap-limited (`U = (w_o+d)/s − T_air`) | `(d − w_p)/s − t₁` | **shrinks** with `s` |

Derivation of the gap-limited width: `(w_o+d)/s − T_air − (w_o+w_p)/s + t₂ = (d−w_p)/s − (T_air − t₂) = (d−w_p)/s − t₁`. The `T_air` does not disappear; it combines with `t₂` and leaves `t₁`, the time the dino spends below `h_o` on the way down, which by symmetry equals the time below `h_o` on the way up. After clearing the obstacle's back edge, the dino still has to descend that last stretch before the next obstacle.

The gap term binds when `d < s·t₂ − w_o`. Otherwise clearance binds.

**Feasibility** (some timing works) requires both:

```
2r/g ≥ (w_o + w_p)/s        (clearable)
d > w_p + s·t₁               (gap large enough)
```

Sanity check: for `h_o = 0`, `t₁ = 0`, the gap condition becomes `d > w_p`.

### 3.4 Caveats on the physics

- **Widths.** Use effective collision widths, i.e. the extents that actually trigger contact.
- **`v₀` may depend on `s`.** The formulas hold for any `v₀`. If the launch speed varies with `s` (or is logged per jump), use the actual value for each jump: `t₁, t₂, T_air` then change with `s`, and the distance covered in the air, `s·T_air`, grows faster than `s` alone.
- **Speed may vary.** If `s` changes over time, use its value at the jump (or `s(t)` over the overlap).
- **Next obstacle type.** The gap constraint assumes you must be on the ground before the next obstacle. Flying obstacles need a clearance *band* (bottom and top heights).

---

## 4. The player model: where reaction time lives

The window answers "which jump times succeed?", which is pure physics. Perception and reaction delay are properties of the **player**, so they do **not** shrink the window. They go into the distribution of `t_j`:

```
t_j = t_vis + R + ε
```

- `t_vis = −D_vis/s`: when the obstacle becomes visible (`D_vis` is the visibility distance).
- `R`: perceive-and-react delay (mean `R̄`, some spread).
- `ε`: motor and timing noise.

Consequences:

1. **The mean `μ(s)` absorbs the delay.** `D_vis/s` shrinks with `s` while `R̄` is fixed, so the player's slack shrinks and `μ` drifts relative to the window. This is a *bias* effect, not only a variance effect.
2. **The spread `σ(s)` absorbs the variability.** `Var(t_j) = Var(R) + Var(ε)`, and both can grow with `s` (less information gathered before deciding).
3. **The support is truncated.** If lead time `D_vis/s` is shorter than the minimum possible `R`, the player cannot jump in time. Encode this with a lower truncation `t_vis + R_min`, or give the model a feature like `lead time − R_min`.

Rather than hand-setting `R`, give the model **lead time** `D_vis/s` as a feature and let `μ(·)`, `σ(·)` be learned. If `μ` drifts with `1/s`, that is the reaction-time signature, and `R̄` can be read from the intercept.

### Does a faster game hurt a noisy player?

With a **fixed** `σ`, in the clearance-limited regime the model says a faster game is *easier*, because `W` grows with `s`. Real players find it harder, and the sources are:

1. **`σ` grows with `s`** (less time to perceive and react). Likely the main cause.
2. **The gap term binds more often.** `W_gap = (d − w_p)/s − t₁` only grows with `s` if `d` grows faster than `s`. If gaps do not keep pace with speed, the gap-limited window shrinks.
3. **`μ` drifts** (the reaction-time effect above).
4. **Space vs. time noise.** A timing error `δ` shifts landing by `s·δ`. If the player judges "jump when the obstacle is `X` pixels away", noise is naturally in distance, `σ_t = σ_x/s`, which *shrinks* with `s`. Which of time or distance is the natural noise unit is empirical (Section 8).

So a fixed-σ Gaussian cannot reproduce "harder when faster". It needs `σ(s)` and/or `μ(s)`.

---

## 5. Methods for modeling `F` (the jump-time distribution)

All of them plug into `P = F(U) − F(L)`.

### 5.1 Gaussian (baseline)

`t_j ~ N(t_n + μ, σ²)`, with `t_n` the intended jump time, `μ` a fixed bias, `σ` a fixed spread.

```
P = Φ((U − t_n − μ)/σ) − Φ((L − t_n − μ)/σ)
```

The best intended time centers the window and cancels the bias, `t_n* = (L+U)/2 − μ`, giving

```
P_max = 2Φ((U − L)/(2σ)) − 1 = 2Φ(W/(2σ)) − 1
```

Simple, but it assumes symmetric, light tails, one bias, and one noise level everywhere.

### 5.2 Heteroscedastic Gaussian (cheapest upgrade)

`μ(x, h)` and `σ(x, h)` come from a regression (linear or boosted). Example: `σ(s) = σ₀(s/s₀)^α` or `σ₀ + k·s`. It captures noise growing with speed and shifts after a close call. `P` now falls with `s` whenever `σ(s)` grows faster than `W`.

### 5.3 Mixture or skewed / heavy-tailed density

Gaussian mixture, skew-normal, or Student-t. Handles occasional very early or very late jumps and asymmetric errors. A mixture component can also represent "did not really try".

### 5.4 Nonparametric

KDE or histograms of the residuals, or **quantile regression** (predict quantiles of `t_j`, read off the CDF at `L` and `U`). No shape assumption, but needs more data.

### 5.5 Discrete-time hazard model

At each frame, predict the probability of jumping *now* given the state. Then `F` follows from the product of "not yet" probabilities. It matches reaction-driven behavior and handles "never jumped in time" naturally (censoring is built in).

### 5.6 Direct classifier (benchmark)

Skip the density: learn `P(success | x, h)` from labeled outcomes (logistic regression or gradient boosting). Easy to build, no need to define an intended jump time, but it has to **learn `L` and `U` from data** and generalizes worse to unseen speeds and obstacle sizes.

Methods 5.1 to 5.5 share the same physics and differ only in the model of `t_j`. Method 5.6 is the odd one out, which is why it works as a check on the others.

---

## 6. A better structure: model the timing, not the outcome

**Why.** Given `t_j`, success is the deterministic window check. A classifier that predicts success directly spends data re-learning physics. Modeling `p(t_j | x, h)` puts the learning where the uncertainty is (the player), and the window shifts by *known* physics when speed, obstacle size, or gap change. That is what makes it generalize.

**Pipeline**

1. Compute `L`, `U` from physics (Section 3), using logged `v₀`, `s`, hitbox widths.
2. Learn `p(t_j | x, h)` with one of 5.1 to 5.5.
3. Predict `P(success) = ∫_L^U p(t_j | x, h) dt_j`.

**Sequential structure.** Two kinds of carryover exist:

- *Physical* (where and when the dino landed, time left to react) is already in the state `x`.
- *Player* (rhythm, adaptation, tendency to jump early or late after a close call) is not. Add lag features: previous jump's timing error `δ_prev`, previous outcome (clean or close call), rolling mean and variance of the last `k` errors, and the previous `t_j` itself.

This is a finite-order Markov approximation, so a recurrent model is rarely needed. Test it: add the lag features and see whether held-out log-likelihood improves. Only go to a GRU if longer histories keep helping.

**Dataset construction (avoid leakage).** Snapshot the state *before* the jump (for example at a fixed reference distance from the obstacle, or per frame). Never include the jump time or anything after it in the features, or the model learns the trivial "jumped at the right time → success".

**Censoring.** Failed runs end the game, so what the player would have done next is never observed, and a missed or never-executed jump has no `t_j`. A hazard model (5.5) handles the "no jump" case directly. One-step-ahead prediction is fine, but multi-step rollouts are biased.

**Role of the classifier (5.6).** Train it alongside as a diagnostic. If the timing model does not beat it on held-out log-loss and calibration, something is wrong. Usual culprits:

- *A misspecified window* (wrong hitboxes, ignored acceleration, wrong gap handling). A classifier quietly absorbs these errors, the hybrid cannot.
- *Unmodeled structure in `t_j`* (for example a "didn't jump" mode that a unimodal density handles badly).

If the classifier wins, find where (which speeds or obstacle types) and fix the window or the density, rather than abandoning the structure.

---

## 7. Evaluating predictions

- **Metrics:** log-loss, Brier score, calibration curves. Not accuracy, since most jumps succeed (heavy class imbalance) and calibration is the point.
- **Splits:** by session or player, not random rows. Consecutive jumps are correlated and random splits look better than they are.
- **Density quality:** held-out log-likelihood of `t_j`, and PIT histograms (below).

---

## 8. Checking whether the Gaussian is bad

First define the residual. Use the error relative to the window center, `δ = t_j − (L+U)/2` (equivalently relative to a fixed reference distance if the player works in space). Compute it in **both time units and distance units** (`s·δ`).

| Check | What to look for |
|---|---|
| **QQ plot / skewness / kurtosis** | heavy tails, skew, multimodality (a "did not try" mode) |
| **`σ` vs `s`** | plot the spread of `δ` against `s`, once in time units and once in distance units. The flatter one is the natural noise unit. If the time-unit spread rises faster than `W`, the intuition "faster is harder" holds in your data |
| **`μ` vs `1/s`** | linear drift suggests a reaction-time component, and the intercept estimates `R̄` |
| **`μ` by obstacle type / height / gap** | systematic shifts mean `μ` should depend on `x` |
| **Autocorrelation of `δ` across consecutive jumps** | near zero means history features will not help much |
| **PIT histogram** of `F(t_j | x, h)` | non-uniform means the model is miscalibrated: U-shaped = too narrow, hump = too wide, skewed = biased |
| **Held-out log-likelihood** | fixed Gaussian vs heteroscedastic vs mixture vs nonparametric |
| **Calibration of `P(success)`** | compare predicted vs observed success rate in bins, especially near the window edges, where the tails matter most |

Rule of thumb: if the heteroscedastic Gaussian fixes the `σ` vs `s` pattern and the PIT is flat, stop there. Move to mixture or hazard models only if tails, multimodality, or "never jumped" cases remain.

---

## 9. Summary

1. Physics gives `L = τ − t₂` and `U = min(−t₁, (w_o+d)/s − T_air)`, with clearability `h_o ≤ v₀²/2g − g·τ²/8`.
2. The window width has two regimes: clearance-limited (grows with `s`) and gap-limited (shrinks with `s`).
3. Reaction and perception belong in the distribution of `t_j` (`μ(s)`, `σ(s)`, truncation), not in the window.
4. `P(success) = F(U) − F(L)`; methods differ only in `F`: Gaussian, heteroscedastic Gaussian, mixture, nonparametric, hazard.
5. Model the timing, not the outcome, and keep a direct classifier as a diagnostic.
6. Check the Gaussian with residual diagnostics in time and distance units, `σ` vs `s`, `μ` vs `1/s`, PIT, and held-out likelihood.
