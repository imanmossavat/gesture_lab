#!/usr/bin/env bash
# One shot: docs/notebooks/data folders + an installable `gesture_lab` package.
# Run from the repo root. Undo everything with:  git switch main && git branch -D package-layout
set -e

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Commit or stash your changes first (git status must be clean)."; exit 1
fi
git switch -c package-layout

mkdir -p gesture_lab docs/images notebooks data/valid
# git mv keeps history; plain mv for files git does not track; skip files that are not there
mv_() { [ -e "$1" ] && { git mv "$1" "$2" 2>/dev/null || mv "$1" "$2"; } || true; }

for m in adaptive analyze analyze_trials dev_bench experiment game gesture_lib jump_service mp_compat progressive report_html simulate_experiment summarize; do
  mv_ "$m.py" gesture_lab/
done
for f in EXPERIMENT.md FUTURE.md DEV.md dino_jump_success_model.md gesture_lab_intentions_and_next_steps.md; do mv_ "$f" docs/; done
mv_ dino_jump_geometry.png docs/images/
mv_ gesture_landmarks_lab.ipynb notebooks/
mv_ data/alice_run1.csv data/valid/

# --- new files -------------------------------------------------------------
echo '"""Gesture Lab: hand-gesture runner that measures how precisely a jump can be timed."""' > gesture_lab/__init__.py

cat > game.py <<'PY'
"""Launcher so `python game.py` keeps working. The code lives in gesture_lab/game.py."""
from gesture_lab.game import main

if __name__ == "__main__":
    main()
PY

cat > pyproject.toml <<'TOML'
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "gesture-lab"
version = "0.1.0"
description = "Hand-gesture runner that measures how precisely a jump can be timed"
requires-python = ">=3.10"
dependencies = [
    "mediapipe>=0.10.0",
    "opencv-python>=4.8.0",
    "pygame>=2.5.0",
    "numpy>=1.26.0",
    "matplotlib>=3.8.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0.0", "jupyter>=1.0.0"]

[tool.setuptools]
packages = ["gesture_lab"]
TOML

cat > requirements.txt <<'REQ'
# Dependencies live in pyproject.toml. This line keeps `pip install -r requirements.txt` working.
-e .[dev]
REQ

printf '\n*.egg-info/\nbuild/\n' >> .gitignore
perl -pi -e 's{^!data/alice_run1\.csv\s*$}{!data/valid/*.csv\n}' .gitignore
grep -q '^!data/valid/\*\.csv' .gitignore || printf '\n!data/valid/*.csv\n' >> .gitignore

# --- rewrite imports, commands and references -------------------------------
python3 - <<'PY'
import pathlib, re

MODS = ("adaptive analyze analyze_trials dev_bench experiment game gesture_lib jump_service "
        "mp_compat progressive report_html simulate_experiment summarize").split()
TOOLS = "analyze analyze_trials dev_bench simulate_experiment summarize".split()
alt, tools = "|".join(MODS), "|".join(TOOLS)
root = pathlib.Path(".")

def edit(path, fn):
    s = path.read_text(encoding="utf-8"); n = fn(s)
    if n != s:
        path.write_text(n, encoding="utf-8")

def imports(s):
    s = re.sub(rf"(?m)^(\s*)from ({alt}) import", r"\1from gesture_lab.\2 import", s)
    return re.sub(rf"(?m)^(\s*)import ({alt})( as \w+)?[ \t]*$", r"\1from gesture_lab import \2\3", s)

def commands(s):          # python analyze_trials.py ...  ->  python -m gesture_lab.analyze_trials ...
    return re.sub(rf"python3? ({tools})\.py", r"python -m gesture_lab.\1", s)

def names(s):             # `adaptive.py`  ->  `gesture_lab/adaptive.py`  (backticked file names in docs)
    return re.sub(rf"`({alt})\.py`", r"`gesture_lab/\1.py`", s)

for p in list(root.glob("gesture_lab/*.py")) + list(root.glob("tests/*.py")):
    edit(p, imports)
for p in root.glob("tests/*.py"):         # tests that run tools as subprocesses
    edit(p, lambda s: re.sub(rf'"({tools})\.py"', r'"-m", "gesture_lab.\1"', s))
for p in list(root.glob("gesture_lab/*.py")) + list(root.glob("tests/*.py")):
    edit(p, commands)
docs = [p for p in [root / "README.md", root / "AGENTS.md"] if p.exists()] + list(root.glob("docs/*.md"))
for p in docs:
    edit(p, lambda s: names(commands(s)))

def readme(s):
    s = s.replace("pip install -r requirements.txt", "pip install -e .       # installs the dependencies and the gesture_lab package")
    s = s.replace("](EXPERIMENT.md)", "](docs/EXPERIMENT.md)").replace("](FUTURE.md)", "](docs/FUTURE.md)")
    s = re.sub(r"(?<![/\w])gesture_landmarks_lab\.ipynb", "notebooks/gesture_landmarks_lab.ipynb", s)
    s = re.sub(r"(?m)^- CSV files are ignored by git;.*$",
               "- CSV files are ignored by git, except in `data/valid/`: put sessions you want to keep there.", s)
    s = s.replace("pytest -q                           ", 'pip install -e ".[dev]"   # adds pytest and jupyter\npytest -q                           ', 1)
    return s
if (root / "README.md").exists():
    edit(root / "README.md", readme)
if (root / "AGENTS.md").exists():
    edit(root / "AGENTS.md", lambda s: s.replace("`EXPERIMENT.md`", "`docs/EXPERIMENT.md`").replace("`FUTURE.md`", "`docs/FUTURE.md`"))
edit(root / "gesture_lab/jump_service.py", lambda s: re.sub(r"(?m)^dino_jump_success_model\.md:", "docs/dino_jump_success_model.md:", s))
edit(root / "gesture_lab/game.py", lambda s: s.replace("(see EXPERIMENT.md)", "(see docs/EXPERIMENT.md)"))
nb = root / "notebooks/gesture_landmarks_lab.ipynb"
if nb.exists():
    edit(nb, lambda s: s.replace('"from gesture_lib import (', '"from gesture_lab.gesture_lib import ('))
wf = root / ".github/workflows/tests.yml"
if wf.exists():
    edit(wf, lambda s: s.replace("pip install -r requirements.txt", 'pip install -e ".[dev]"')
                         .replace("cache: pip", "cache: pip\n          cache-dependency-path: pyproject.toml"))
PY

echo
echo "Done. Next:"
echo '  pip install -e ".[dev]"   # installs the package and dev tools'
echo "  pytest -q"
echo "  git add -A && git commit -m 'Package layout: gesture_lab package, docs/, notebooks/, data/valid/'"
echo "  git switch main && git merge --ff-only package-layout"
