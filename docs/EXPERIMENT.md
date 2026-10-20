# Experiment mode: measuring how precisely a player times a jump

This guide explains what experiment mode measures, how to run it, and how to read the report. It assumes no statistics background.

## 1. The question

The normal game adapts its difficulty to the player with one dial, `d`. That tells us how hard the game became, but not **why**. Experiment mode asks a more specific question:

> How precisely does this person time a jump?

We answer it with a small experiment. The game shows obstacles that are **easy** (you may jump almost anywhere) and obstacles that are **hard** (you must jump at almost exactly the right moment), in a shuffled order. For each obstacle we record when you jumped. Afterwards we measure how scattered your jumps are.

## 2. The safe window

Physics decides which jump times work. Take an obstacle 30 px wide and 87 px tall, with the game running at 400 px/s.

- The player is 42 px wide, so the player and obstacle overlap while the obstacle moves **72 px** past the player. At 400 px/s that is **0.18 s**.
- A jump keeps the player above 87 px for **0.42 s**.
- To clear the obstacle, the player must be above 87 px for the whole 0.18 s overlap. So the jump has to be timed so that this 0.18 s sits inside the 0.42 s the player spends high up. The spare time is `0.42 − 0.18 = 0.24 s`.

That spare time is the **safe window**: you may start your jump anywhere in a 240 ms span. In numbers: any jump that starts between **388 ms and 148 ms before** the obstacle reaches you clears it. Start earlier and you land on the obstacle. Start later and you hit it on the way up.

```
 time before the obstacle reaches you ->   -388 ms ............ -148 ms        0
                                              [ safe window: 240 ms wide ]
                                          too early                  too late
```

The **taller** the obstacle, the less time you spend above it, so the narrower the window. Experiment mode picks obstacle heights that give windows from 450 ms (easy) down to 60 ms (hard, you must be within +-30 ms).

Terms used throughout:

| Term | Meaning |
|---|---|
| `t = 0` | the moment the obstacle's front edge reaches the front edge of the player |
| `t_j` | when you started your jump, on that clock (negative = before the obstacle arrived) |
| `L`, `U` | earliest and latest `t_j` that clear the obstacle |
| `W = U - L` | the width of the safe window |
| `delta` | `t_j` minus the middle of the window. 0 = perfectly centred, negative = early, positive = late |
| `delta_open` | Like `delta`, but for when the camera first *saw* your gesture (minus half a camera frame): your timing with the game's own wait removed. Bias and spread are computed from this. Empty in older logs |
| `t_open`, `open_lag_ms`, `cam_interval_ms` | The pieces behind `delta_open` |

## 3. Running it

```bash
# Play it (camera needed). Takes about 70 seconds.
python game.py --experiment --log-trials data/alice_trials.csv --player alice

# Read the result
python -m gesture_lab.analyze_trials data/alice_trials.csv --plot data/alice_report.png
```

What happens in the game:

1. 3 warm-up obstacles with the widest window, to settle in.
2. 28 more obstacles: 4 shuffled blocks, each block using every window width once. No more than 2 hard obstacles in a row, and the same width never twice in a row.
3. The game speed stays constant (400 px/s by default), only the heights change, and the time between obstacles varies randomly so you cannot tap in a rhythm.
4. Crashes do not stop the run. When all obstacles are played the screen shows "EXPERIMENT COMPLETE".

Useful options: `--experiment-speed 350`, `--experiment-blocks 6` (longer, more precise), `--experiment-seed 3` (repeat the same order).

No camera? `python -m gesture_lab.simulate_experiment --mu 0.03 --sigma 0.05 --out sim.csv` plays the same experiment with a simulated player whose bias (`--mu`) and spread (`--sigma`) you choose. Run the analysis on it and see whether the numbers come back. That is a good way to learn what the report looks like.

## 4. What is recorded

One row per obstacle, in `--log-trials`:

| Column | Meaning |
|---|---|
| `trial` | obstacle number in this run |
| `target_w` | the window width the obstacle was designed for (s) |
| `height`, `width`, `speed` | the obstacle and game speed (px, px, px/s) |
| `L`, `U`, `W` | the safe window from the physics (s) |
| `t_j` | when you jumped (s); empty if you never jumped in time |
| `delta` | `t_j` minus the middle of the window (s) |
| `delta_open` | Like `delta`, but for when the camera first *saw* your gesture (minus half a camera frame): your timing with the game's own wait removed. Bias and spread are computed from this. Empty in older logs |
| `t_open`, `open_lag_ms`, `cam_interval_ms` | The pieces behind `delta_open` |
| `predicted_success` | 1 if `t_j` was inside `[L, U]`: what the physics predicts |
| `outcome` | 1 if the game says you cleared it, 0 if you crashed |
| `failure_mode` | for a crash: `early`, `late`, `no_jump`, or `unexplained` |

Two columns that look the same are deliberately separate: `predicted_success` comes from our physics and `outcome` comes from the game. Comparing them tests whether the physics is right (section 5.2).

## 5. Reading the report

`gesture_lab/analyze_trials.py` prints five numbered sections. Each ends with a short "What this means".

### 5.1 What was recorded
How many obstacles you cleared, and **why** crashes happened. Too early and too late are timing errors. "No jump" means no jump started before the obstacle arrived.

### 5.2 Does the physics predict the game?
For every jump, was the physics right about the result? A few disagreements right at a window edge are normal, because the game moves in 1/60 s steps and one frame can decide. If many disagreements are far from an edge, something in the physics is wrong (sizes, jump speed, frame rate) and the rest of the report should not be trusted until that is fixed.

### 5.3 Your timing: bias and spread
Two numbers describe your jumps:

- **Bias**: your average `delta_open` (`delta` in older logs). "+30 ms" means you jump 30 ms later than the middle of the window, on average.
- **Spread**: how scattered your jumps are around your own average (the standard deviation). "50 ms" means about two thirds of your jumps land within 50 ms of your average.

Bias is a habit you could correct. Spread is the precision you cannot easily remove, and it decides how narrow a window you can handle. The report shows how uncertain each number is. With 31 obstacles the spread is only known to about +-6 ms.

### 5.4 Success against window width
A table of how often you cleared obstacles of each window width, next to what a bell-curve model predicts from your bias and spread alone. If the model were perfect, the observed rates would match the prediction within the "95% range". With only a handful of obstacles per row, that range is wide: luck matters a lot, which is why one short session says little.

### 5.5 Is a bell curve a reasonable model?
The prediction in 5.4 assumes your timing errors follow a **bell curve** (a Gaussian). This section checks that assumption instead of trusting it:

- **Skewness**: are errors lopsided (more late than early)? 0 is symmetric.
- **Kurtosis**: are there more extreme jumps than a bell curve allows? 0 is a bell curve.
- **QQ plot**: a picture where a bell curve gives a straight line.
- **Same spread for all windows?** A simple model assumes narrow-window (tall) obstacles are as easy to time as wide ones. If the spread differs, the model should depend on obstacle height.
- **Warm-up or tiring?** Compares the first and second half of the session, and says whether the difference is bigger than chance would give.

### The picture (`--plot`)

| Panel | What to look for |
|---|---|
| Top left: success against window width | Red curve (the model) passing through the blue points (what happened) |
| Top right: where your jumps landed | Bars centred on 0 and shaped like the red bell curve |
| Bottom left: QQ plot | Dots on the straight line |
| Bottom right: every obstacle in order | Each dot is a jump. Grey bar is that obstacle's safe zone. Dots outside the bar are red (crashes) |

## 6. What can go wrong with the measurement

- **Few obstacles.** About 30 obstacles gives rough numbers. For a precise measurement, play two or three runs and pass all files to `gesture_lab/analyze_trials.py` together.
- **Frames.** A jump can only start on a frame boundary (every 17 ms), so measured bias runs about 8 ms late. The window itself is corrected for the game's frame-by-frame update (see the comments in `gesture_lab/experiment.py`).
- **No jump.** If you never jump in time there is no timing to measure, so those obstacles are left out of bias and spread. A player who misses a lot looks slightly better than they are.
- **Learning.** People improve during a session. Section 5.5 checks this.
- **Gesture delay.** The camera and gesture recognition add their own delay (see `--log-latency`). Your measured bias includes it: bias = your habit + system delay.

## 7. Where things live

| File | Role |
|---|---|
| `gesture_lab/jump_service.py` | the physics: safe window `[L, U]`, window to height, timing models |
| `gesture_lab/experiment.py` | the obstacle plan (`make_trials`), the recorder, the frame corrections |
| `gesture_lab/game.py` | `--experiment` and `--log-trials` |
| `gesture_lab/simulate_experiment.py` | a simulated player, no camera |
| `gesture_lab/analyze_trials.py` | the report and the picture |
| `tests/test_experiment.py`, `tests/test_analyze_trials.py` | what the code is expected to do |

The derivation of the window is in `dino_jump_success_model.md`.

## 8. Not done yet

This first version measures **timing precision only**. Not yet measured: how short a gap between obstacles you can handle, how speed changes your timing, and how late an obstacle can appear. Each can be added the same way: a new list of designed obstacles and one more analysis.


## Two clocks: when the game acted, and when the camera saw you

`t_j` / `delta` say when the **game** started the jump. That is what decides clear or crash, so
the physics check and the "model says" column use it.

`t_open` / `delta_open` say when the **camera first saw your gesture**, minus half a camera frame
(the average wait for the next frame). It removes the game's own wait: the frames the gesture
must be held (stability) and the average sampling delay. Bias and spread come from it.

What is still inside `delta_open`: the camera's internal delay (exposure, processing, USB), which
software cannot see, and your own movement time until the pose is recognised. Neither can be
corrected here. Measure the camera delay once with a slow-motion video of your hand and the
screen if you need absolute numbers. When comparing gestures, compare spreads rather than biases.
