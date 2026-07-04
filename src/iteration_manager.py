#!/usr/bin/env python3
"""
ITERATION MANAGER
=================
Deferred retry of medium and bad pools after iteration 1 completes.

Retry rounds:
  Round 1: Medium pool → strategy 2 (few_shot)
  Round 2: Bad pool    → strategy 15 (full_pipeline)

Both rounds use the ACCUMULATED good pool at retry time — richer context
than was available during the original generation attempt. This is the
core "deferred" mechanism: waiting until the pool is maximally rich
before retrying.
"""

from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

from parser import preprocess_problem


class IterationManager:

    def __init__(self, generator, quality_evaluator):
        self.gen  = generator
        self.eval = quality_evaluator

    # ── internal ──────────────────────────────────────────────────────────────

    def _retry_one(
        self,
        exemplar     : Dict,
        good_exemplars: List[Dict],
        strategy_id  : int,
        manual_exemplars: List[Dict],
        categorizer,
        knn_fn       = None,
    ) -> Tuple[Dict, str]:
        """
        Re-generate for a single problem and evaluate with REx metric.
        Returns (enriched_exemplar_dict, new_category).
        """
        question     = exemplar.get('question', exemplar.get('problem', ''))
        ground_truth = exemplar.get('ground_truth',
                       exemplar.get('raw_answer',
                       exemplar.get('answer', '')))

        cat_info = categorizer.categorize(question)

        # Preprocess for REx evaluator (needs clean_answer + computations)
        problem_dict = preprocess_problem({
            'question': question,
            'answer'  : ground_truth,
        })

        gen = self.gen.generate(
            problem          = question,
            strategy_id      = strategy_id,
            manual_exemplars = manual_exemplars,
            pool_exemplars   = good_exemplars,
            category_info    = cat_info,
            return_logprobs  = True,
            knn_fn           = knn_fn,
        )

        qual = self.eval.evaluate(gen['trace'], problem_dict)

        enriched = {
            **exemplar,
            'trace'        : gen['trace'],
            'pred_answer'  : qual['pred_answer'],
            'gold_answer'  : qual['gold_answer'],
            'rex_score'    : qual['rex_score'],
            'validity'     : qual['validity'],
            'execution'    : qual['execution'],
            'redundancy'   : qual['redundancy'],
            'is_correct'   : qual['is_correct'],
            'num_steps'    : qual['num_steps'],
            'n_gold_ops'   : qual['n_gold_ops'],
            'category'     : qual['category'],
            'strategy_id'  : strategy_id,
            'strategy_name': gen['strategy_name'],
            'retry'        : True,
        }
        return enriched, qual['category']

    # ── public ────────────────────────────────────────────────────────────────

    def retry_medium_pool(
        self,
        medium_pool                   : List[Dict],
        good_exemplars                : List[Dict],
        strategy_id                   : int = 2,
        manual_exemplars_for_strategy : List[Dict] = None,
        quality_ev                    = None,
        categorizer                   = None,
        knn_fn                        = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Retry medium pool with accumulated good exemplars.
        Recovery rule: reclassified as GOOD → move to good pool.

        Returns: (recovered_good, still_medium_or_bad)
        """
        if quality_ev:
            self.eval = quality_ev

        manual_exemplars_for_strategy = manual_exemplars_for_strategy or []
        recovered = []
        remaining = []

        print(f"\n[Retry MEDIUM] {len(medium_pool)} problems | strategy={strategy_id}")

        for ex in tqdm(medium_pool, desc='medium retry'):
            try:
                enriched, new_cat = self._retry_one(
                    ex, good_exemplars, strategy_id,
                    manual_exemplars_for_strategy,
                    categorizer, knn_fn,
                )
                enriched['recovered_from'] = 'medium'
                if new_cat == 'good':
                    recovered.append(enriched)
                else:
                    remaining.append(ex)
            except Exception as e:
                print(f"  [warn] medium retry: {e}")
                remaining.append(ex)

        n = len(recovered) + len(remaining)
        print(f"  Recovered {len(recovered)}/{n} "
              f"({100*len(recovered)/max(1,n):.1f}%) from MEDIUM")
        return recovered, remaining

    def retry_bad_pool(
        self,
        bad_pool                      : List[Dict],
        good_exemplars                : List[Dict],
        strategy_id                   : int = 15,
        manual_exemplars_for_strategy : List[Dict] = None,
        quality_ev                    = None,
        categorizer                   = None,
        knn_fn                        = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Retry bad pool with accumulated good exemplars.
        Recovery rule: reclassified as GOOD or MEDIUM → recovered.
        (bad → medium is still useful as a retry candidate)

        Returns: (recovered_good_or_medium, still_bad)
        """
        if quality_ev:
            self.eval = quality_ev

        manual_exemplars_for_strategy = manual_exemplars_for_strategy or []
        recovered = []
        remaining = []

        print(f"\n[Retry BAD] {len(bad_pool)} problems | strategy={strategy_id}")

        for ex in tqdm(bad_pool, desc='bad retry'):
            try:
                enriched, new_cat = self._retry_one(
                    ex, good_exemplars, strategy_id,
                    manual_exemplars_for_strategy,
                    categorizer, knn_fn,
                )
                enriched['recovered_from'] = 'bad'
                if new_cat in ('good', 'medium'):
                    recovered.append(enriched)
                else:
                    remaining.append(ex)
            except Exception as e:
                print(f"  [warn] bad retry: {e}")
                remaining.append(ex)

        n = len(recovered) + len(remaining)
        print(f"  Recovered {len(recovered)}/{n} "
              f"({100*len(recovered)/max(1,n):.1f}%) from BAD")
        return recovered, remaining