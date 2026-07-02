#!/usr/bin/env python3
"""
QUALITY EVALUATOR  —  REx Score
=================================
Research basis:
  ReasonEval (Xia et al., AAAI 2025 oral) outperforms ROSCOE on MR-GSM8K
  by a massive margin: ROSCOE-SA achieves 52.5 F1 vs ReasonEval at 81.4–90.9
  AUC. ROSCOE is embedding-based and "does not provide evaluations comparable
  across different models" (Putnam-AXIOM benchmark, 2025).

Our metric — REx Score (Reasoning Execution Score):
  REx = 0.40·V + 0.35·E + 0.25·R

  V = Step Validity      (are the arithmetic steps actually correct?)
  E = Execution Fidelity (does the trace reproduce GSM8K gold computations?)
  R = Redundancy Penalty (are steps concise and non-repetitive?)

Why this is better than our previous metric (F/I/K/A):
  - F (ROSCOE faithfulness) and I (ROSCOE informativeness): both shown to
    have <55% F1 on MR-GSM8K vs humans — unreliable for our use case
  - K (token log-prob confidence): a model can be confidently wrong
  - A (arithmetic check): we keep this but now compute it PROPERLY using
    the GSM8K gold <<expr=result>> annotations as the ground truth
    when available, falling back to regex-only otherwise

New components:
  V — Step Validity:
      For each step in the generated trace, verify every explicit arithmetic
      expression (A op B = C). Count fraction that are numerically correct.
      If a step has no explicit expression: check if any number in it matches
      a gold computation result (partial credit).
      This is the core idea from ReasonEval's "validity" dimension, implemented
      without needing a fine-tuned judge model.

  E — Execution Fidelity (new, GSM8K-specific):
      The GSM8K human solutions contain <<expr=result>> annotations for
      every computation. After preprocessing, we have a list of gold
      computations for each problem. E measures what fraction of those
      gold computation results appear (correctly) in the generated trace.
      This is a direct, verifiable signal that the model is actually
      executing the right steps — unique to GSM8K, not possible on MATH500.

  R — Redundancy (adapted from ReasonEval's "redundancy" dimension):
      Measures step efficiency: penalises traces that repeat information
      or include steps that don't advance the solution.
      Computed as: 1 - (fraction of steps that are near-duplicates of
      a previous step). Near-duplicate = normalized text edit distance < 0.3.
      Score 1.0 = no redundancy, 0.0 = all steps are repeats.

Classification (unchanged — empirically reasonable thresholds):
  CORRECT + REx >= 0.65  → GOOD
  CORRECT + 0.35 <= REx < 0.65 → MEDIUM
  CORRECT + REx < 0.35   → BAD
  WRONG   + REx >= 0.65  → MEDIUM  (good process, wrong execution)
  WRONG   + REx < 0.65   → BAD
"""

import re
import math
from typing import List, Optional, Tuple, Dict

from parser import (
    extract_answer,
    answers_match,
    extract_computations,
    parse_solution_steps,
    normalize_number,
)


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

GOOD_THRESHOLD   = 0.65
MEDIUM_THRESHOLD = 0.35


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT V — Step Validity
# ─────────────────────────────────────────────────────────────────────────────

_EXPR_RE = re.compile(
    r'(-?[\d,]+(?:\.\d+)?)\s*([+\-\*/×÷])\s*(-?[\d,]+(?:\.\d+)?)\s*=\s*(-?[\d,]+(?:\.\d+)?)'
)

_OP_MAP = {
    '+' : lambda a, b: a + b,
    '-' : lambda a, b: a - b,
    '*' : lambda a, b: a * b,
    '×' : lambda a, b: a * b,
    '/' : lambda a, b: a / b if b != 0 else float('inf'),
    '÷' : lambda a, b: a / b if b != 0 else float('inf'),
}

def _check_expression(a_s, op, b_s, c_s) -> bool:
    try:
        a = float(a_s.replace(',', ''))
        b = float(b_s.replace(',', ''))
        c = float(c_s.replace(',', ''))
        expected = _OP_MAP.get(op, lambda x, y: None)(a, b)
        if expected is None:
            return False
        tol = max(1e-6, abs(expected) * 1e-4)
        return abs(c - expected) <= tol
    except Exception:
        return False


def score_step_validity(trace: str, gold_computations: List[Tuple[str, str]] = None) -> float:
    """
    V — Step Validity score [0, 1].

    Primary: check all explicit A op B = C expressions in the trace.
    Secondary: if gold_computations provided, also check that the model's
               numeric results match the gold computation results.

    Returns:
        1.0 if all verifiable expressions are correct
        0.4 if no verifiable expressions found (neutral, not penalised)
        fraction correct otherwise
    """
    # Primary: regex-based expression check
    exprs = _EXPR_RE.findall(trace)
    if exprs:
        correct = sum(1 for a, op, b, c in exprs if _check_expression(a, op, b, c))
        primary_score = correct / len(exprs)
    else:
        primary_score = None   # no expressions to check

    # Secondary: gold computation result coverage
    if gold_computations:
        gold_results = set()
        for expr_str, result_str in gold_computations:
            n = normalize_number(result_str)
            if n:
                gold_results.add(n)

        if gold_results:
            # Count how many gold results appear correctly in the trace
            trace_numbers = set()
            for raw_num in re.findall(r'-?[\d,]+(?:\.\d+)?', trace):
                n = normalize_number(raw_num)
                if n:
                    trace_numbers.add(n)
            hits = len(gold_results & trace_numbers)
            secondary_score = hits / len(gold_results)
        else:
            secondary_score = None
    else:
        secondary_score = None

    # Combine
    if primary_score is not None and secondary_score is not None:
        return round(0.60 * primary_score + 0.40 * secondary_score, 4)
    elif primary_score is not None:
        return round(primary_score, 4)
    elif secondary_score is not None:
        return round(secondary_score, 4)
    else:
        return 0.40   # neutral: no verifiable content found


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT E — Execution Fidelity (GSM8K-specific, uses gold <<>> annotations)
# ─────────────────────────────────────────────────────────────────────────────

def score_execution_fidelity(trace: str,
                              gold_computations: List[Tuple[str, str]]) -> float:
    """
    E — Execution Fidelity score [0, 1].

    Measures how many of the GOLD computation results from the GSM8K
    human solution appear in the generated trace.

    This is a direct, verifiable signal unique to GSM8K:
      - Perfect trace: contains all intermediate results from the gold solution
      - Partial trace: gets some but not all intermediate values right
      - Wrong trace  : intermediate numbers don't match gold at all

    If no gold computations available (e.g. for model-generated exemplars
    that don't have <<>> annotations), falls back to 0.5 (neutral).
    """
    if not gold_computations:
        return 0.50   # neutral — can't verify without gold

    gold_results = []
    for _, result_str in gold_computations:
        n = normalize_number(result_str)
        if n:
            gold_results.append(n)

    if not gold_results:
        return 0.50

    # Extract all numbers from the generated trace
    trace_numbers = set()
    for raw in re.findall(r'-?[\d,]+(?:\.\d+)?', trace):
        n = normalize_number(raw)
        if n:
            trace_numbers.add(n)

    hits = sum(1 for gr in gold_results if gr in trace_numbers)
    return round(hits / len(gold_results), 4)


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT R — Redundancy Score (adapted from ReasonEval)
# ─────────────────────────────────────────────────────────────────────────────

def _normalized_edit_distance(s1: str, s2: str) -> float:
    """Normalized Levenshtein distance [0, 1] between two strings."""
    if not s1 and not s2:
        return 0.0
    if not s1 or not s2:
        return 1.0
    # Use character-level edit distance
    m, n = len(s1), len(s2)
    # For very long strings, use quick approximation
    if m > 200 or n > 200:
        # Jaccard similarity on word sets as proxy
        w1, w2 = set(s1.lower().split()), set(s2.lower().split())
        if not w1 and not w2:
            return 0.0
        jaccard = len(w1 & w2) / len(w1 | w2)
        return 1.0 - jaccard

    # Full DP edit distance
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n] / max(m, n)


def score_redundancy(trace: str) -> float:
    """
    R — Redundancy score [0, 1].  1.0 = no redundancy (ideal), 0.0 = all steps repeated.

    A step is "redundant" (near-duplicate) if its normalized edit distance
    to any PREVIOUS step is < 0.30 (i.e. >70% similar to a prior step).
    Score = 1 - (n_redundant_steps / total_steps)

    Edge cases:
      - 0 or 1 steps: return 0.8 (can't have redundancy, but also can't
        verify quality — mild penalty for being too short)
      - All unique steps: return 1.0
    """
    steps = [s.strip() for s in trace.split('\n') if s.strip()
             and not s.strip().startswith('####')]

    if len(steps) <= 1:
        return 0.80   # too short to evaluate, mild penalty

    redundant = 0
    for i in range(1, len(steps)):
        for j in range(i):
            dist = _normalized_edit_distance(steps[i], steps[j])
            if dist < 0.30:   # >70% similar = near-duplicate
                redundant += 1
                break   # count each step at most once as redundant

    return round(1.0 - (redundant / len(steps)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify(is_correct: bool, rex: float) -> str:
    if is_correct:
        if rex >= GOOD_THRESHOLD:   return 'good'
        if rex >= MEDIUM_THRESHOLD: return 'medium'
        return 'bad'
    else:
        if rex >= GOOD_THRESHOLD:   return 'medium'
        return 'bad'


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC CLASS
# ─────────────────────────────────────────────────────────────────────────────

class QualityEvaluator:
    """
    Evaluate a generated reasoning trace against a preprocessed problem.

    REx Score = 0.40·V + 0.35·E + 0.25·R

    Usage:
        ev = QualityEvaluator()

        # With full preprocessing (best — uses gold computations):
        result = ev.evaluate(
            trace        = gen['trace'],
            problem      = preprocessed_problem,   # dict from parser.preprocess_problem()
        )

        # Minimal usage (no gold computations — E falls back to 0.5):
        result = ev.evaluate(
            trace        = gen['trace'],
            problem      = {'clean_answer': '42', 'computations': [], 'expected_steps': 3},
        )
    """

    good_threshold   = GOOD_THRESHOLD
    medium_threshold = MEDIUM_THRESHOLD

    def evaluate(self, trace: str, problem: Dict) -> Dict:
        """
        Args:
            trace   : model-generated reasoning trace string
            problem : preprocessed problem dict (from parser.preprocess_problem())
                      Must have at minimum 'clean_answer' key.
                      Ideally also has 'computations' and 'expected_steps'.

        Returns:
            {
                'rex_score'    : float  — overall REx score [0,1]
                'validity'     : float  — V component [0,1]
                'execution'    : float  — E component [0,1]
                'redundancy'   : float  — R component [0,1]
                'is_correct'   : bool
                'pred_answer'  : str | None
                'gold_answer'  : str | None
                'category'     : 'good' | 'medium' | 'bad'
                'num_steps'    : int    — steps in generated trace
                'n_gold_ops'   : int    — gold computation count
                'ops_covered'  : float  — fraction of gold ops in trace
            }
        """
        gold_answer   = problem.get('clean_answer')
        gold_comps    = problem.get('computations', [])
        expected_steps = problem.get('expected_steps', 0)

        pred_answer = extract_answer(trace)
        is_correct  = answers_match(pred_answer, gold_answer)

        V = score_step_validity(trace, gold_comps)
        E = score_execution_fidelity(trace, gold_comps)
        R = score_redundancy(trace)

        rex = round(0.40 * V + 0.35 * E + 0.25 * R, 4)
        cat = classify(is_correct, rex)

        # Step count in generated trace
        gen_steps = [s for s in trace.split('\n')
                     if s.strip() and not s.strip().startswith('####')]

        return {
            'rex_score'   : rex,
            'validity'    : V,
            'execution'   : E,
            'redundancy'  : R,
            'is_correct'  : is_correct,
            'pred_answer' : pred_answer,
            'gold_answer' : gold_answer,
            'category'    : cat,
            'num_steps'   : len(gen_steps),
            'n_gold_ops'  : len(gold_comps),
            'ops_covered' : E,
        }