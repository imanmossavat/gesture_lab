# Future work

Ideas and research context that are **not built yet**. What exists is in `README.md` and `EXPERIMENT.md`.

## Where this fits

The core model here, "timing error is roughly Gaussian and a jump succeeds if it lands inside the safe window", is the standard idea in *temporal pointing* research (hitting a moving target at the right moment). Starting points from a short search; read them before citing:

- Modelling error rates in temporal pointing (CHI 2016): https://displaydaily.com/wp-content/uploads/2016/06/modeling-errors-chi2016.pdf
- Moving target selection: a cue integration model (CHI 2018): https://acris.aalto.fi/ws/portalfiles/portal/21506957/ELEC_Lee_et_al_Moving_target_CHI2018.pdf
- User performance in consecutive temporal pointing: https://dgist.elsevierpure.com/en/publications/user-performance-in-consecutive-temporal-pointing-an-exploratory-/
- The tradeoff between spatial jitter and latency in pointing tasks: https://vvise.iat.sfu.ca/pubs/pavlovych2009tradeoff
- In-air un-instrumented pointing performance: https://yorkspace.library.yorku.ca/items/f0554c6f-d5c7-432a-8971-c3f27d5b27f5

The gap this lab could fill: temporal precision of **camera-tracked hand gestures**, with windows defined by game physics, across gestures and speeds. I found no such study in two quick searches, but that is not proof none exists. Check properly before claiming novelty.

## Questions the lab can already answer

1. How does timing spread depend on the jump gesture (open hand vs thumb up)?
2. How does it depend on speed? (`--progressive` gives a threshold window per speed.)
3. Are failures Gaussian timing errors, or Gaussian plus misfires and misses?
4. Does bias drift within a session? (The HTML report has a drift model.)
5. Engagement: needs a survey step and a randomized manipulation (see below).

**Sizing a study.** The uncertainty on a spread is about σ / √(2n). To see a 15 ms difference in spread between two gestures for one player (σ about 70 ms) you need about 100 jumps per gesture (n ≈ (σ / SE)², with SE of the difference about 7 ms). Run `gesture_lab/simulate_experiment.py` first to check a design before anyone plays. A sensible first study: within players, counterbalanced gesture order, 8 to 10 players, about 3 sessions per gesture.

## Methods beyond what exists

- **Hierarchical Bayesian model.** Each player gets a bias and spread drawn from a population. With about 30 jumps per player it gives steadier estimates than separate fits, and it is how to test a gesture effect across players (PyMC, Stan or brms).
- **Lapse rate.** Model misfires and misses as a separate probability so a few extreme early jumps stop inflating the spread. Standard in psychophysics; the third session suggested it (crashes on the widest windows, heavy tails).
- **Model comparison.** Gaussian vs Student-t vs Gaussian plus lapses, compared by leave-one-out cross-validation instead of reading a QQ plot by eye.
- **Bayesian adaptive design.** Choose the next obstacle height to maximize information about bias and spread (QUEST, the Psi method, expected information gain). `gesture_lab/progressive.py` is a simple staircase; this would be its principled upgrade.
- **Counterbalancing.** Randomize condition order across players (Latin square) so learning is not confused with a gesture effect.
- **Effective window.** Report 4.13 x spread (the convention in pointing studies) so gestures compare fairly across target widths.
- **Simulation-based power analysis.** Simulate the whole study, including the planned model, before collecting data.
- **Drift beyond one session.** The Kalman drift model currently runs per file; run it over pooled sessions with a session effect.

The drift model's behavior (from a simulation of 200 sessions each, 31 jumps): it flagged drift in about 1.5% of sessions with a flat bias, in 87% of clear drifts at 40 ms jump noise, but only in about 42% at 60 ms noise. A "no clear drift" result is therefore weak evidence at this session length.

## Gameplay ideas

- **A director with patterns.** Singles, tight doubles (where the gap becomes the binding bound), staircases of heights, rests. Give each pattern a cost, its expected failures under the player's fitted model, `P(clear) = Phi((W/2 - mu)/sigma) - Phi((-W/2 - mu)/sigma)`, and run a tension curve (build, spike, release) that spends a budget. Variety comes from sequencing, not from a better single difficulty number.
- **Calibrate, play, re-measure.** A short staircase estimates the threshold, a play phase varies patterns around it, a final block re-measures.
- **More actions.** A second gesture to duck with low obstacles; more poses. Log **every** raw gesture event (not only jumps near obstacles) so misses and misfires per gesture can be counted, and record each gesture's stability setting in the log header, since a pose that must hold m frames adds delay.
- **Feedback as an experiment.** Show the HTML report to half the players and compare their next session (knowledge of results may change timing).

## Collecting data from other people

- A pseudonymous player ID, a consent line, and 2 to 3 survey items (enjoyment, fairness, wish to continue) saved with the session.
- A data README for the CSV columns, a `data/valid/` folder, a reproducible notebook, and a Zenodo release for a citable snapshot.
- If participants are not just friends, check whether ethics approval is needed. Logs contain no camera images; keep it that way.

## Known gaps

- **Thumb gesture** is tuned on idealized synthetic hands. It needs a check with real hands (orientation, handedness, different thumbs); `THUMB_RISE_MIN` in `gesture_lab/gesture_lib.py` is the knob.
- **Progressive mode** is verified only with a scripted player; it needs a pilot with people to see whether it feels less repetitive than the fixed list.
- **Delay still in the bias.** `delta_open` removes the stability wait and the average sampling wait. Still in it: the camera's internal delay and your movement time until the pose is recognised. A slow-motion video calibration (240 fps phone, hand and screen in one shot) would measure the camera part. Log the measured camera FPS in the trial header.
- **Old logs.** Sessions recorded before commit `0901def` have wrong windows for tall obstacles; do not pool them.
