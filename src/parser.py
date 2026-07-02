#!/usr/bin/env python3
"""
PARSER + PREPROCESSOR  —  GSM8K
=================================
Two separate concerns kept in one file:

  1. DATASET PREPROCESSING  (run once, before any generation)
     Cleans the raw GSM8K answer field into structured fields:
       - clean_answer   : normalised numeric string  e.g. "276"
       - steps          : list of text steps (split on \n)
       - computations   : list of (expr, result) from <<a+b=c>> annotations
       - expected_steps : integer count of solution steps

  2. ANSWER EXTRACTION  (run on every model-generated trace)
     Robust pipeline matching the field used in ReasonEval / LFM2 / standard GSM8K evals:
       Priority 1 – #### <number>
       Priority 2 – "The answer is X" / "answer: X"
       Priority 3 – Last number in the text (fallback)
     + full normalization: strip commas, currency, units, convert floats

Both are referenced by quality_evaluator.py and main.py / Kaggle notebook.
"""

import re
from typing import Optional, List, Tuple, Dict


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# GSM8K inline computation annotation: <<left_side=right_side>>
_COMPUTATION_RE = re.compile(r'<<([^<>]+)=([^<>]+)>>')

# Final answer marker
_HASH_MARKER_RE = re.compile(r'####\s*([\d,\.\-]+)')

# Natural language answer phrases
_ANSWER_PHRASE_RE = re.compile(
    r'(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:|was|will\s+be)\s*([\d,\.\-]+)',
    re.IGNORECASE
)

# Any number (for fallback)
_ANY_NUMBER_RE = re.compile(r'-?[\d,]+(?:\.\d+)?')

# Units and currency to strip during normalization
_UNITS_RE = re.compile(
    r'\s*(?:dollars?|cents?|euros?|pounds?|yuan|rupees?|'
    r'hours?|minutes?|seconds?|days?|weeks?|months?|years?|'
    r'miles?|km|meters?|feet|inches?|cm|'
    r'kg|grams?|pounds?|ounces?|'
    r'liters?|gallons?|ml|'
    r'percent|%|'
    r'items?|units?|pieces?|each|times?|students?|people|persons?|'
    r'books?|apples?|oranges?|cars?|boxes?|bags?|pairs?|'
    r'tickets?|coins?|marbles?|balls?|flowers?|trees?)\b',
    re.IGNORECASE
)
_CURRENCY_RE = re.compile(r'[\$£€¥₹]')


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize_number(text: str) -> Optional[str]:
    """
    Normalize a raw numeric string extracted from model output or ground truth.

    Pipeline (in order):
      1. Strip currency symbols ($, £, €, ¥, ₹)
      2. Strip common unit words (dollars, hours, students, ...)
      3. Remove commas used as thousands separators  (1,024 → 1024)
      4. Strip leading/trailing whitespace
      5. Convert to float, then back to string to canonicalize
         (removes trailing zeros: "18.0" → "18", keeps "18.5" as "18.5")
      6. Return None if nothing parseable remains

    NOTE: We do NOT strip units blindly before checking context — e.g.
    if the answer is "18 dollars" we strip "dollars" to get "18", which is
    correct. But if the answer is "18 cents" we'd also get "18", which is
    wrong if the GT is "0.18". This is a known limitation of unit-stripping
    (documented in parser literature). For GSM8K specifically, all GT answers
    are plain integers or simple decimals, so this is safe.
    """
    if not text:
        return None

    s = str(text).strip()

    # Step 1-2: remove currency and units
    s = _CURRENCY_RE.sub('', s)
    s = _UNITS_RE.sub('', s)

    # Step 3: remove commas (thousands separator)
    s = s.replace(',', '')

    # Step 4: strip whitespace
    s = s.strip()

    if not s:
        return None

    # Step 5: parse as float to canonicalize, then format
    try:
        val = float(s)
        # Return as integer string if whole number, else keep decimals
        if val == int(val) and '.' not in s:
            return str(int(val))
        elif val == int(val):
            return str(int(val))
        else:
            # Round to 4 dp to avoid float noise, then strip trailing zeros
            formatted = f'{val:.4f}'.rstrip('0').rstrip('.')
            return formatted
    except (ValueError, OverflowError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ANSWER EXTRACTION  (from model-generated traces)
# ─────────────────────────────────────────────────────────────────────────────

def extract_answer(text: str) -> Optional[str]:
    """
    Extract and normalize the final answer from a model-generated trace.

    Priority order (matching standard GSM8K eval practice):
      1. #### <number>             — canonical GSM8K marker
      2. "the answer is X"         — natural language answer phrase
      3. Last number in text       — fallback (most papers use this)

    Returns a normalized numeric string, or None if nothing found.
    """
    if not text:
        return None

    # Priority 1: #### marker
    m = _HASH_MARKER_RE.search(text)
    if m:
        return normalize_number(m.group(1))

    # Priority 2: natural language phrase
    m = _ANSWER_PHRASE_RE.search(text)
    if m:
        return normalize_number(m.group(1))

    # Priority 3: last number in text
    nums = _ANY_NUMBER_RE.findall(text)
    if nums:
        return normalize_number(nums[-1])

    return None


def answers_match(pred: Optional[str], gold: Optional[str],
                  tolerance: float = 1e-9) -> bool:
    """
    Compare two normalized answer strings with float tolerance.
    Handles None safely. Uses tolerance=1e-9 (matches SLMJury / standard practice).
    """
    if pred is None or gold is None:
        return False
    if pred == gold:
        return True
    try:
        return abs(float(pred) - float(gold)) <= tolerance
    except (ValueError, TypeError):
        return pred.strip().lower() == gold.strip().lower()


# ─────────────────────────────────────────────────────────────────────────────
# DATASET PREPROCESSING  (run once on the GSM8K training set)
# ─────────────────────────────────────────────────────────────────────────────

def extract_computations(solution_text: str) -> List[Tuple[str, str]]:
    """
    Extract all <<expression=result>> computation annotations from a
    GSM8K human-written solution.

    These are the gold-standard arithmetic steps in GSM8K — every
    computation the human solver performed is annotated inline.
    Returns list of (expression_str, result_str) tuples.

    Example:
      "John has <<3*4=12>> 12 apples" → [("3*4", "12")]
    """
    return _COMPUTATION_RE.findall(solution_text)


def parse_solution_steps(solution_text: str) -> List[str]:
    """
    Split a GSM8K human-written solution into clean steps.

    Steps in GSM8K are separated by newlines. We:
      1. Remove the #### final answer line (it's not a reasoning step)
      2. Remove <<...>> computation annotations inline (keep surrounding text)
      3. Strip empty lines
    """
    # Remove #### line
    lines = [l for l in solution_text.split('\n')
             if not l.strip().startswith('####')]

    # Remove <<...>> annotations but keep surrounding text
    cleaned = []
    for line in lines:
        line = _COMPUTATION_RE.sub(lambda m: f'= {m.group(2)}', line)
        line = line.strip()
        if line:
            cleaned.append(line)

    return cleaned


def preprocess_problem(raw: Dict) -> Dict:
    """
    Preprocess a single raw GSM8K problem dict.

    Input (from HuggingFace datasets):
      {'question': str, 'answer': str}
      where 'answer' contains the full solution ending with \n#### <number>

    Output:
      {
        'question'        : str,   original question (unchanged)
        'raw_answer'      : str,   full solution text (unchanged)
        'clean_answer'    : str,   normalized final number e.g. "276"
        'steps'           : list,  reasoning steps (text, no #### line)
        'computations'    : list,  [(expr, result), ...] from <<>> annotations
        'expected_steps'  : int,   number of reasoning steps
        'n_computations'  : int,   number of gold arithmetic operations
      }
    """
    question   = raw.get('question', '')
    answer_raw = raw.get('answer', '')

    # Extract final numeric answer
    final_ans = extract_answer(answer_raw)

    # Extract step-level content
    steps        = parse_solution_steps(answer_raw)
    computations = extract_computations(answer_raw)

    return {
        'question'       : question,
        'raw_answer'     : answer_raw,
        'clean_answer'   : final_ans,
        'steps'          : steps,
        'computations'   : computations,
        'expected_steps' : len(steps),
        'n_computations' : len(computations),
    }


def preprocess_dataset(dataset, max_problems: int = None) -> List[Dict]:
    """
    Preprocess a full GSM8K split (HuggingFace dataset object).
    Returns list of preprocessed problem dicts.

    Usage:
        from datasets import load_dataset
        ds = load_dataset('openai/gsm8k', 'main')
        train = preprocess_dataset(ds['train'])
        test  = preprocess_dataset(ds['test'])
    """
    n = min(len(dataset), max_problems) if max_problems else len(dataset)
    processed = []
    for i in range(n):
        processed.append(preprocess_problem(dataset[i]))
    return processed