# temp.md: state of the docs (delete this file after the cleanup)

**Status: transitional.** The documentation is currently a mix of useful explanations, fast-moving facts that go stale, and files that are not documentation at all. This file says which is which until the cleanup is done. After that, delete it.

## For agents working in this repo right now

1. Read `AGENTS.md` and this file. Nothing else under `*.md` is required reading.
2. Do **not** treat any other `.md` file as the source of truth for numbers, defaults, flags or file lists. The code, the tests and `--help` win. When they disagree with a doc, the doc is wrong.
3. Do not read the `.patch` files. They are not documentation, and one is already out of date.
4. Do not add new `.md` files. Do not add numbers, option defaults or file lists to existing docs; point to where they live instead.

## Rules for the cleanup

- Docs explain **why**: the idea, the reasoning, the trade-off. They do not repeat what the code says.
- Anything that changes often lives only in code, tests or generated output: constants (`Config`), CLI options (`--help`), measured numbers (`dev_bench.py report`), the list of files.
- Two audiences, kept apart: **humans** (learning, deciding) and **agents** (invariants and commands to work safely). A file serves one of them.

## What each file is today

| File | Audience | What it is | Problem | Proposed action |
|---|---|---|---|---|
| `AGENTS.md` | agents | File map, invariants, commands | The file map is a list that goes stale as files are added | Keep invariants and commands. Cut the file map down to one line per area, or drop it |
| `README.md` | humans | Quick start, controls, data commands | Repeats options and example commands that `--help` already gives | Keep the purpose and the first run. Link to the rest instead of copying it |
| `DEV.md` | humans (developers) | How the difficulty, physics and spawn rhythm work | Mixes the reasoning with tuned numbers ("about 1.9 s at d = 1", "sd 0.4-0.6 s", "never below 0.44 s") | Keep the reasoning. Move the numbers to `dev_bench.py report` |
| `EXPERIMENT.md` | humans (learning) | Educational guide to experiment mode and its report | The worked example uses today's constants (87 px, 0.18 s, 388 ms); defaults like "31 obstacles, 70 s" can change | Keep it as the "why and how to read the result" guide. Mark the example as illustrative or regenerate it |
| `dino_jump_success_model.md` | humans (theory) | Derivation of the jump window and the player models | Stable reasoning, but written for the textbook physics. The game's frame-by-frame corrections are only in `experiment.py` | Keep. Add a short note on the frame corrections, or link to `game_physics` |
| `gesture_lab_intentions_and_next_steps.md` | humans | Plan: intentions, requirements, next steps | A planning document. Steps 1 to 3 are done, so it is already stale | Move the "intention" paragraph into `README.md`. Delete the rest |
| `experiment_mode.patch` | nobody | A patch of the experiment-mode change | Not documentation. No longer applies to the current code. The change is already in the files | Delete |
| `wild_spawn.patch` | nobody | A patch of the spawn-rhythm change | Not documentation. The change is already in git history | Delete |
| `temp.md` | agents, humans | This file | Exists only to explain the mess | Delete after the cleanup |

## Open decisions

- **Where do human docs live?** A `docs/` folder would let `AGENTS.md` say "do not read `docs/` unless asked", which is the clearest way to separate the audiences. Today everything is in the repo root.
- **Where do the experiment-mode defaults live?** The obstacle list, speed and window widths are code constants in `experiment.py`. `EXPERIMENT.md` should point there and not repeat them.
- **Is `jump_service.py` code or documentation?** It is code. Its docstrings carry the derivation, so `dino_jump_success_model.md` could shrink to the reasoning only.

## Done means

- `AGENTS.md` has no list that grows with every new file.
- No `.md` file contains a number that `Config`, `--help` or a test can produce.
- No `.patch` files in the repo.
- `temp.md` deleted.
