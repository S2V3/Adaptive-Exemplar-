#!/usr/bin/env python3
"""
EXEMPLAR POOL
=============
Manages three pools: good / medium / bad.

Key features:
  • Stores seed exemplars separately so they're always available
  • Dynamic retrieval: get K exemplars most similar to a query by category_type
  • Sorted by quality_score descending within each pool
  • Save / load JSON for Kaggle ↔ local round-trips
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


class ExemplarPool:
    """
    Thread-unsafe (not needed here) in-memory store for three quality pools.
    """

    def __init__(self):
        self._good   : List[Dict] = []
        self._medium : List[Dict] = []
        self._bad    : List[Dict] = []

    # ── properties (read-only lists) ─────────────────────────
    @property
    def good_pool(self)   -> List[Dict]: return self._good
    @property
    def medium_pool(self) -> List[Dict]: return self._medium
    @property
    def bad_pool(self)    -> List[Dict]: return self._bad

    # Allow iteration manager to replace medium/bad lists after retry
    @medium_pool.setter
    def medium_pool(self, val): self._medium = val
    @bad_pool.setter
    def bad_pool(self,   val): self._bad   = val

    # ── add / move ────────────────────────────────────────────
    def add(self, exemplar: Dict, category: str):
        """Add one exemplar to the appropriate pool."""
        if category == 'good':
            self._good.append(exemplar)
        elif category == 'medium':
            self._medium.append(exemplar)
        else:
            self._bad.append(exemplar)

    def add_batch(self, exemplars: List[Dict], category: str):
        for ex in exemplars:
            self.add(ex, category)

    def move_to_good(self, exemplar: Dict):
        """Move a recovered exemplar directly into the good pool."""
        self._good.append(exemplar)

    # ── retrieval ─────────────────────────────────────────────
    def get_good_exemplars(
        self,
        k: int = 5,
        category_type: Optional[str] = None,
        strategy: str = 'top_score',
    ) -> List[Dict]:
        """
        Retrieve up to K good exemplars.

        Args:
            k             : max exemplars to return
            category_type : if given, prefer exemplars matching this type;
                            fills remaining slots with any-type exemplars
            strategy      : 'top_score'  → sort by hybrid_score desc
                            'diverse'    → one per category_type then fill
                            'random'     → random sample (useful for SC)
        """
        pool = sorted(
            self._good,
            key=lambda x: x.get('hybrid_score', x.get('quality_score', 0)),
            reverse=True,
        )

        if strategy == 'random':
            return random.sample(pool, min(k, len(pool)))

        if category_type:
            # Priority: same-category first
            same = [e for e in pool if e.get('category_type') == category_type]
            diff = [e for e in pool if e.get('category_type') != category_type]
            pool = same + diff

        if strategy == 'diverse':
            # One exemplar per seen category, then fill
            seen_cats: set = set()
            diverse = []
            rest    = []
            for ex in pool:
                cat = ex.get('category_type', 'other')
                if cat not in seen_cats:
                    diverse.append(ex)
                    seen_cats.add(cat)
                else:
                    rest.append(ex)
            pool = diverse + rest

        return pool[:k]

    # ── statistics ────────────────────────────────────────────
    def stats(self) -> Dict:
        total = len(self._good) + len(self._medium) + len(self._bad)
        def pct(n): return round(100 * n / total, 1) if total else 0.0

        avg_score = (
            sum(e.get('hybrid_score', 0) for e in self._good) / len(self._good)
            if self._good else 0.0
        )

        # Category breakdown in good pool
        cat_counts: Dict[str, int] = {}
        for ex in self._good:
            cat = ex.get('category_type', 'unknown')
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        return {
            'total'              : total,
            'good'               : len(self._good),
            'medium'             : len(self._medium),
            'bad'                : len(self._bad),
            'good_pct'           : pct(len(self._good)),
            'medium_pct'         : pct(len(self._medium)),
            'bad_pct'            : pct(len(self._bad)),
            'avg_good_score'     : round(avg_score, 4),
            'good_by_category'   : cat_counts,
        }

    def summary(self):
        s = self.stats()
        print(f"\n{'='*60}")
        print(f"  EXEMPLAR POOL SUMMARY")
        print(f"{'='*60}")
        print(f"  Good   : {s['good']:4d}  ({s['good_pct']}%)  avg_score={s['avg_good_score']}")
        print(f"  Medium : {s['medium']:4d}  ({s['medium_pct']}%)")
        print(f"  Bad    : {s['bad']:4d}  ({s['bad_pct']}%)")
        print(f"  Total  : {s['total']:4d}")
        if s['good_by_category']:
            print(f"\n  Good pool by category:")
            for cat, n in sorted(s['good_by_category'].items(), key=lambda x: -x[1]):
                print(f"    {cat:<15} : {n}")
        print(f"{'='*60}\n")

    # ── persistence ───────────────────────────────────────────
    def save(self, output_dir: str = '.', prefix: str = ''):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag    = f"_{prefix}_{ts}" if prefix else f"_{ts}"

        for pool_name, data in [
            ('good',   self._good),
            ('medium', self._medium),
            ('bad',    self._bad),
        ]:
            path = out / f"{pool_name}_pool{tag}.json"
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"  Saved {len(data):4d} {pool_name} exemplars → {path}")

    @classmethod
    def load(cls, good_path: str, medium_path: str = None, bad_path: str = None):
        """Re-hydrate an ExemplarPool from saved JSON files."""
        pool = cls()

        def _load(path, pool_name):
            if path and Path(path).exists():
                with open(path) as f:
                    return json.load(f)
            else:
                print(f"  [warn] {pool_name} pool file not found: {path}")
                return []

        pool._good   = _load(good_path,   'good')
        pool._medium = _load(medium_path, 'medium') if medium_path else []
        pool._bad    = _load(bad_path,    'bad')    if bad_path    else []

        print(f"  Loaded pool: {len(pool._good)} good, "
              f"{len(pool._medium)} medium, {len(pool._bad)} bad")
        return pool