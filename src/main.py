#!/usr/bin/env python3
"""
MAIN — LOCAL SETUP
==================
Run this on your laptop (no GPU needed).

What it does:
  1. Loads GSM8K from HuggingFace
  2. Preprocesses the first 300 problems (extracts gold computations,
     clean answers, step counts) using parser.py
  3. Runs ExemplarSelector on those 300 — embeds, scores, categorises,
     selects n-per-category exemplars for each of 16 strategies
  4. Saves outputs_local/manual_exemplars.json
  5. Saves outputs_local/remaining_problems.json
  6. Saves outputs_local/kaggle_config.json
  7. Prints category and strategy summary

After this runs:
  git add outputs_local/ src/
  git commit -m "exemplar selection done"
  git push
  → then run notebook/main_kaggle.py on Kaggle

Run from project root:
  cd C:\\Research\\Adaptive-exemplar
  python src/main.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Works whether run as script or notebook cell
sys.path.insert(0, str(Path.cwd() / 'src'))

from datasets          import load_dataset
from parser            import preprocess_dataset
from categorizer       import QuestionCategorizer
from exemplar_selector import ExemplarSelector, STRATEGY_EXEMPLAR_NEEDS
from generation        import STRATEGY_NAMES


# ── Load dataset ────────────────────────────────────────────────────────────────

print("Loading GSM8K ...")
ds       = load_dataset('openai/gsm8k', 'main')
train_ds = ds['train']
test_ds  = ds['test']
print(f"  Train : {len(train_ds)}")
print(f"  Test  : {len(test_ds)}  (held out — never touch until final eval)")


# ── Config ─────────────────────────────────────────────────────────────────────

POOL_SIZE = len(train_ds)    # problems used for exemplar selection (never used for generation)
PILOT_SIZE = 200    # max problems to include in remaining_problems.json for Kaggle pilot
OUTPUT_DIR = Path('./outputs_local')
OUTPUT_DIR.mkdir(exist_ok=True)


# ── STEP 1: Preprocess first 300 problems ──────────────────────────────────────
# parser.preprocess_problem() extracts:
#   clean_answer   → normalized final number e.g. "11"
#   computations   → gold <<expr=result>> pairs e.g. [("3*5","15"), ("15-4","11")]
#   steps          → list of reasoning steps
#   expected_steps → int
#   n_computations → int
# These fields are needed by quality_evaluator.py on Kaggle.

print(f"\n{'='*65}")
print(f"STEP 1 — Preprocessing first {POOL_SIZE} problems")
print(f"{'='*65}")

import random
random.seed(42)   # fixed seed so results are reproducible
pool_raw = preprocess_dataset(train_ds, max_problems=POOL_SIZE)
print(f"  Preprocessed {len(pool_raw)} problems")

# Quick sanity check — show one example
sample = pool_raw[0]
print(f"\n  Sample problem (index 0):")
print(f"    question      : {sample['question'][:70]}...")
print(f"    clean_answer  : {sample['clean_answer']}")
print(f"    computations  : {sample['computations'][:2]}")
print(f"    expected_steps: {sample['expected_steps']}")
print(f"    n_computations: {sample['n_computations']}")


# ── STEP 2: Select exemplars ────────────────────────────────────────────────────

print(f"\n{'='*65}")
print(f"STEP 2 — Exemplar Selection (n per category per strategy)")
print(f"{'='*65}")

categorizer = QuestionCategorizer()
selector    = ExemplarSelector(pool_size=POOL_SIZE)

manual_exemplars, remaining_problems = selector.select(
    pool_raw,
    categorizer,
    ground_truth_key='raw_answer',   # use raw_answer so Kaggle can re-preprocess
)

print(f"\n  Total exemplars selected : {sum(len(v) for v in manual_exemplars.values())}")
print(f"  Remaining for generation : {len(remaining_problems)}")


# ── STEP 3: Save outputs ────────────────────────────────────────────────────────

print(f"\n{'='*65}")
print(f"STEP 3 — Saving outputs")
print(f"{'='*65}")

# manual_exemplars: keys are strategy IDs (int → str for JSON)
with open(OUTPUT_DIR / 'manual_exemplars.json', 'w') as f:
    json.dump({str(k): v for k, v in manual_exemplars.items()}, f, indent=2)
print(f"  ✓ manual_exemplars.json")


# Randomly sample 200 for the pilot — fixed seed for reproducibility
pilot_problems = random.sample(remaining_problems, min(200, len(remaining_problems)))

with open(OUTPUT_DIR / 'remaining_problems.json', 'w') as f:
    json.dump(pilot_problems, f, indent=2)

with open(OUTPUT_DIR / 'remaining_problems_full.json', 'w') as f:
    json.dump(remaining_problems, f, indent=2)

print(f"  ✓ remaining_problems.json        ({len(pilot_problems)} problems — random pilot sample)")
print(f"  ✓ remaining_problems_full.json   ({len(remaining_problems)} problems — full pool for Phase 2)")


# Verify zero overlap between exemplars and remaining problems
exemplar_questions = set()
for exs in manual_exemplars.values():
    for ex in exs:
        exemplar_questions.add(ex['question'])

overlap = sum(1 for p in remaining_problems if p['question'] in exemplar_questions)
print(f"  Overlap check: {overlap} problems appear in both exemplars and remaining")
print(f"  {'✓ No overlap' if overlap == 0 else '! WARNING: overlap found'}")

# kaggle_config: tells Kaggle notebook what settings were used here
kaggle_config = {
    'pool_size'               : POOL_SIZE,
    'pilot_size'              : PILOT_SIZE,
    'total_remaining'         : len(remaining_problems),
    'num_strategies'          : 16,
    'strategy_names'          : STRATEGY_NAMES,
    'strategy_exemplar_needs' : STRATEGY_EXEMPLAR_NEEDS,
    'quality_metric'          : 'REx = 0.40·V + 0.35·E + 0.25·R',
    'good_threshold'          : 0.65,
    'medium_threshold'        : 0.35,
}
with open(OUTPUT_DIR / 'kaggle_config.json', 'w') as f:
    json.dump(kaggle_config, f, indent=2)
print(f"  ✓ kaggle_config.json")


# ── STEP 4: Category analysis ───────────────────────────────────────────────────

print(f"\n{'='*65}")
print(f"STEP 4 — Category Analysis")
print(f"{'='*65}")

# Category distribution in selected exemplars
exemplar_cats: dict = defaultdict(int)
for exs in manual_exemplars.values():
    for ex in exs:
        exemplar_cats[ex.get('category_type', 'unknown')] += 1

print("\n  Selected exemplars by category:")
for cat, n in sorted(exemplar_cats.items(), key=lambda x: -x[1]):
    print(f"    {cat:<15} {n:3d}  {'█' * min(n, 50)}")

# Category distribution in remaining problems
remaining_cats: dict = defaultdict(int)
for p in remaining_problems:
    remaining_cats[p.get('category_type', 'unknown')] += 1

print("\n  Remaining problems by category (these go to Kaggle):")
for cat, n in sorted(remaining_cats.items(), key=lambda x: -x[1]):
    print(f"    {cat:<15} {n:3d}  {'█' * min(n, 40)}")


# ── STEP 5: Strategy summary ────────────────────────────────────────────────────

print(f"\n{'='*65}")
print(f"STEP 5 — Strategy Exemplar Summary")
print(f"{'='*65}")
print(f"\n  {'ID':>3}  {'Strategy':<25}  {'n/cat':>5}  {'Selected':>8}  {'Status'}")
print(f"  {'-'*55}")

all_ok = True
for sid in range(16):
    name     = STRATEGY_NAMES[sid]
    n_per_cat= STRATEGY_EXEMPLAR_NEEDS[sid]
    selected = len(manual_exemplars.get(sid, []))
    expected = n_per_cat * 8   # 8 category types
    ok       = selected >= expected or n_per_cat == 0
    status   = '✓' if ok else f'! expected {expected}'
    if not ok:
        all_ok = False
    print(f"  {sid:>3}  {name:<25}  {n_per_cat:>5}  {selected:>8}  {status}")

if all_ok:
    print(f"\n  ✓ All strategies have sufficient exemplars")
else:
    print(f"\n  ! Some strategies are short on exemplars")
    print(f"    This can happen if the 300-problem pool has few problems")
    print(f"    in some category types. Check category distribution above.")


# ── STEP 6: Preprocessing stats ────────────────────────────────────────────────

print(f"\n{'='*65}")
print(f"STEP 6 — Preprocessing Statistics")
print(f"{'='*65}")

total_comps = sum(p.get('n_computations', 0) for p in pool_raw)
total_steps = sum(p.get('expected_steps', 0) for p in pool_raw)
avg_comps   = total_comps / len(pool_raw)
avg_steps   = total_steps / len(pool_raw)

print(f"\n  Problems preprocessed      : {len(pool_raw)}")
print(f"  Total gold computations    : {total_comps}")
print(f"  Avg computations / problem : {avg_comps:.1f}")
print(f"  Avg steps / problem        : {avg_steps:.1f}")
print(f"\n  These gold computations are used by quality_evaluator.py")
print(f"  on Kaggle for the Execution Fidelity (E) component of REx.")


# ── STEP 7: Pipeline overview and next steps ────────────────────────────────────

print(f"""
{'='*65}
PIPELINE OVERVIEW
{'='*65}

GSM8K Train (7473 problems)
  │
  ├─ First {POOL_SIZE} → preprocess → ExemplarSelector
  │                          │
  │                          ├─ embed (sentence-transformers)
  │                          ├─ score (richness heuristic)
  │                          ├─ categorize (8 types)
  │                          └─ select n-per-category per strategy
  │                               └─ manual_exemplars.json  ← pushed to GitHub
  │
  └─ Remaining {len(remaining_problems)} → remaining_problems.json  ← pushed to GitHub

KAGGLE (pulls from GitHub, runs with LLaMA-2-7B on GPU)
  │
  ├─ ITERATION 1: {PILOT_SIZE} problems, round-robin 16 strategies
  │     generate() → evaluate() → REx = 0.40·V + 0.35·E + 0.25·R
  │     → GOOD / MEDIUM / BAD
  │     → good traces added to pool immediately (adaptive context)
  │
  ├─ DEFERRED RETRY 1: medium pool → strategy 2 (few_shot)
  │     uses full accumulated good pool (seeds + iter1 good)
  │
  ├─ DEFERRED RETRY 2: bad pool → strategy 15 (full_pipeline)
  │     uses full accumulated good pool
  │
  └─ SAVE + VISUALIZE → download from Kaggle Output tab

{'='*65}
NEXT STEPS
{'='*65}

  1. Push to GitHub:
       git add src/ outputs_local/ notebook/ README.md .gitignore requirements.txt
       git commit -m "exemplar selection complete, ready for Kaggle"
       git push

  2. On Kaggle:
       Cell 1: !git clone https://github.com/YOUR_USERNAME/adaptive-exemplar /kaggle/working/repo
       Then paste cells from notebook/main_kaggle.py in order

  3. After Kaggle run:
       Download outputs/ folder
       Fill in papers/notes/experiment_log.md with results
""")

print("✓ LOCAL SETUP COMPLETE")