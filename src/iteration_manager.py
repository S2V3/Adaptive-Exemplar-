#!/usr/bin/env python3
"""
ITERATION MANAGER
=================
Handles deferred retry logic for medium and bad pools.

Design:
  • After ITERATION 1: medium + bad pools are retried
  • Retry uses ACCUMULATED good exemplars (seed + recovered from iter 1)
  • Medium pool: retry with few_shot (strategy 2)
  • Bad pool:    retry with full_pipeline (strategy 15) — bigger guns
  • Recovery rule:
      medium → good if category == 'good'
      bad    → good if category in {'good', 'medium'}
"""

from tqdm import tqdm
from typing import List, Dict, Tuple

from quality_evaluator import QualityEvaluator
from generation import ExemplarGenerator


class IterationManager:

    def __init__(
        self,
        generator: ExemplarGenerator,
        quality_evaluator: QualityEvaluator,
    ):
        self.gen  = generator
        self.eval = quality_evaluator

    # ── internal ──────────────────────────────────────────────
    def _retry_one(
        self,
        exemplar: Dict,
        good_exemplars: List[Dict],
        strategy_id: int,
    ) -> Tuple[Dict, str]:
        """
        Re-generate for a single problem and evaluate.
        Returns (enriched_exemplar, new_category).
        """
        question     = exemplar.get('question', exemplar.get('problem', ''))
        ground_truth = exemplar.get('ground_truth', exemplar.get('answer', ''))
        cat_info     = {
            k: exemplar[k]
            for k in ('category_type', 'complexity', 'main_operation')
            if k in exemplar
        }

        result = self.gen.generate(
            problem      = question,
            strategy_id  = strategy_id,
            exemplars    = good_exemplars,
            category_info= cat_info or None,
        )

        quality = self.eval.evaluate(result['trace'], ground_truth)

        enriched = {
            **exemplar,                         # keep all existing fields
            'trace'          : result['trace'],
            'predicted_answer': quality['predicted_answer'],
            'hybrid_score'   : quality['hybrid_score'],
            'semantic_score' : quality['semantic_score'],
            'validity_score' : quality['validity_score'],
            'is_correct'     : quality['is_correct'],
            'strategy_id'    : strategy_id,
            'strategy_name'  : result['strategy_name'],
            'retry'          : True,
        }
        return enriched, quality['category']

    # ── public ────────────────────────────────────────────────
    def retry_medium_pool(
        self,
        medium_pool: List[Dict],
        good_exemplars: List[Dict],
        strategy_id: int = 2,         # few_shot
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Retry medium pool with accumulated good exemplars.

        Returns:
            (recovered_good, still_medium_or_bad)
        """
        recovered = []
        remaining = []

        print(f"\n[Retry MEDIUM] {len(medium_pool)} problems | strategy={strategy_id}")
        for ex in tqdm(medium_pool, desc='medium retry'):
            try:
                enriched, new_cat = self._retry_one(ex, good_exemplars, strategy_id)
                enriched['recovered_from'] = 'medium'
                if new_cat == 'good':
                    recovered.append(enriched)
                else:
                    remaining.append(ex)   # keep original if not recovered
            except Exception as e:
                print(f"  [warn] medium retry error: {e}")
                remaining.append(ex)

        n_total = len(recovered) + len(remaining)
        pct = 100 * len(recovered) / n_total if n_total else 0
        print(f"  Recovered {len(recovered)}/{n_total} ({pct:.1f}%) from MEDIUM\n")
        return recovered, remaining

    def retry_bad_pool(
        self,
        bad_pool: List[Dict],
        good_exemplars: List[Dict],
        strategy_id: int = 15,        # full_pipeline — best chance for bad problems
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Retry bad pool.  A problem is recovered even if it ends up 'medium'.

        Returns:
            (recovered_good_or_medium, still_bad)
        """
        recovered = []
        remaining = []

        print(f"\n[Retry BAD] {len(bad_pool)} problems | strategy={strategy_id}")
        for ex in tqdm(bad_pool, desc='bad retry'):
            try:
                enriched, new_cat = self._retry_one(ex, good_exemplars, strategy_id)
                enriched['recovered_from'] = 'bad'
                if new_cat in ('good', 'medium'):
                    recovered.append(enriched)
                else:
                    remaining.append(ex)
            except Exception as e:
                print(f"  [warn] bad retry error: {e}")
                remaining.append(ex)

        n_total = len(recovered) + len(remaining)
        pct = 100 * len(recovered) / n_total if n_total else 0
        print(f"  Recovered {len(recovered)}/{n_total} ({pct:.1f}%) from BAD\n")
        return recovered, remaining