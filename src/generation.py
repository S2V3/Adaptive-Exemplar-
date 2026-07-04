#!/usr/bin/env python3
"""
GENERATION MODULE
Input:  Problem + Strategy ID (0-15) + Exemplar Pool
Output: {trace, answer, strategy_name, num_tokens}

16 Strategies across 5 groups:
  GROUP 1 PROMPT-BASED     : 0 zero_shot, 1 zero_shot_cot, 2 few_shot, 3 few_shot_dynamic
  GROUP 2 SAMPLING-BASED   : 4 self_consistency, 5 temp_low, 6 temp_high
  GROUP 3 FORMATTING-BASED : 7 explicit_steps, 8 scratchpad, 9 math_first
  GROUP 4 CONTEXT-BASED    : 10 category_context, 11 worked_example, 12 reversed_qa
  GROUP 5 HYBRID           : 13 cot_few_shot, 14 cot_self_consistency, 15 full_pipeline
"""

import re
import torch
from collections import Counter
from typing import List, Dict, Optional, Tuple


# ─────────────────────────────────────────────
# Utility: parse final answer from a trace
# ─────────────────────────────────────────────
def extract_final_answer(text: str) -> Optional[str]:
    """
    Shared answer-extraction parser (used by generator & evaluator).
    Priority order:
        1. GSM8K canonical marker  #### <number>
        2. "the answer is X" / "answer: X"
        3. Last bare number in the text
    """
    # 1. canonical marker
    m = re.search(r'####\s*([\d,]+(?:\.\d+)?)', text)
    if m:
        return m.group(1).replace(',', '').strip()

    # 2. natural-language answer phrase
    m = re.search(
        r'(?:the\s+)?answer\s*(?:is|=|:)\s*([\d,]+(?:\.\d+)?)',
        text, re.IGNORECASE
    )
    if m:
        return m.group(1).replace(',', '').strip()

    # 3. last number fallback
    nums = re.findall(r'[\d,]+(?:\.\d+)?', text)
    if nums:
        return nums[-1].replace(',', '').strip()

    return None


# ─────────────────────────────────────────────
# Majority-vote helper for self-consistency
# ─────────────────────────────────────────────
def majority_vote(answers: List[Optional[str]]) -> Optional[str]:
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


# ─────────────────────────────────────────────
# Prompt-builder helpers
# ─────────────────────────────────────────────
_SYSTEM_HEADER = (
    "You are an expert math tutor. Solve the following grade-school math "
    "problem step by step and end with #### <answer>.\n\n"
)

def _fmt_exemplars(exemplars: List[Dict], n: int = 3) -> str:
    """Format n exemplars as Q/A blocks."""
    out = ""
    for i, ex in enumerate(exemplars[:n], 1):
        out += f"Example {i}:\nQuestion: {ex['question']}\nSolution: {ex['trace']}\n\n"
    return out

def _fmt_category_hint(category_info: Dict) -> str:
    if not category_info:
        return ""
    cat  = category_info.get('category_type', 'unknown')
    comp = category_info.get('complexity', 'unknown')
    return (
        f"[Problem type: {cat.upper()} | Complexity: {comp}]\n"
        "Keep this context in mind while solving.\n\n"
    )


# ─────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────
class ExemplarGenerator:
    """Generate reasoning trajectories using one of 16 strategies."""

    STRATEGY_NAMES = {
        0:  'zero_shot',
        1:  'zero_shot_cot',
        2:  'few_shot',
        3:  'few_shot_dynamic',
        4:  'self_consistency',
        5:  'temp_low',
        6:  'temp_high',
        7:  'explicit_steps',
        8:  'scratchpad',
        9:  'math_first',
        10: 'category_context',
        11: 'worked_example',
        12: 'reversed_qa',
        13: 'cot_few_shot',
        14: 'cot_self_consistency',
        15: 'full_pipeline',
    }

    def __init__(self, model=None, tokenizer=None):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = (
            next(model.parameters()).device if model else
            ('cuda' if torch.cuda.is_available() else 'cpu')
        )

    # ── low-level generation ──────────────────
    def _generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Run model.generate() and return a list of decoded strings
        (one per sequence), stripped of the prompt.
        """
        enc = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)
        prompt_len = enc['input_ids'].shape[1]

        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=(temperature > 0),
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        results = []
        for seq in out:
            # Decode only the newly generated tokens
            new_tokens = seq[prompt_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append(text.strip())
        return results

    # ── prompt builders (one per strategy) ───
    def _build_prompt(
        self,
        strategy_id: int,
        problem: str,
        exemplars: List[Dict],
        category_info: Optional[Dict] = None,
    ) -> Tuple[str, dict]:
        """
        Returns (prompt_string, generation_kwargs_overrides).
        generation_kwargs_overrides can override temperature / num_sequences.
        """
        gen_kwargs = {}   # overrides for this strategy

        # ── GROUP 1: PROMPT-BASED ──────────────
        if strategy_id == 0:
            # Zero-Shot baseline: just instruction + problem
            prompt = (
                f"{_SYSTEM_HEADER}"
                f"Question: {problem}\n"
                f"Answer:"
            )

        elif strategy_id == 1:
            # Zero-Shot + CoT
            prompt = (
                f"{_SYSTEM_HEADER}"
                f"Question: {problem}\n"
                f"Let's think step by step.\n"
            )

        elif strategy_id == 2:
            # Few-Shot (best 3 from pool)
            prompt = (
                f"{_SYSTEM_HEADER}"
                f"{_fmt_exemplars(exemplars, 3)}"
                f"Question: {problem}\n"
                f"Solution:"
            )

        elif strategy_id == 3:
            # Few-Shot + Dynamic (most similar exemplars by type)
            # Similarity is handled externally; we receive pre-ranked exemplars
            prompt = (
                f"{_SYSTEM_HEADER}"
                f"Here are similar problems and their solutions:\n\n"
                f"{_fmt_exemplars(exemplars, 3)}"
                f"Now solve this problem the same way:\n"
                f"Question: {problem}\n"
                f"Solution:"
            )

        # ── GROUP 2: SAMPLING-BASED ────────────
        elif strategy_id == 4:
            # Self-Consistency: 5 generations, majority vote
            prompt = (
                f"{_SYSTEM_HEADER}"
                f"Question: {problem}\n"
                f"Let's think step by step.\n"
            )
            gen_kwargs['num_return_sequences'] = 5
            gen_kwargs['temperature'] = 0.9   # slightly higher for diversity

        elif strategy_id == 5:
            # Low Temperature (0.3) — conservative
            prompt = (
                f"{_SYSTEM_HEADER}"
                f"Question: {problem}\n"
                f"Let's think step by step.\n"
            )
            gen_kwargs['temperature'] = 0.3

        elif strategy_id == 6:
            # High Temperature (1.2) — creative
            prompt = (
                f"{_SYSTEM_HEADER}"
                f"Question: {problem}\n"
                f"Let's think step by step.\n"
            )
            gen_kwargs['temperature'] = 1.2

        # ── GROUP 3: FORMATTING-BASED ──────────
        elif strategy_id == 7:
            # Explicit Step Format
            prompt = (
                f"{_SYSTEM_HEADER}"
                f"Solve the following problem using numbered steps.\n\n"
                f"Question: {problem}\n\n"
                f"Step 1:"
            )

        elif strategy_id == 8:
            # Scratchpad Format
            prompt = (
                f"{_SYSTEM_HEADER}"
                f"Question: {problem}\n\n"
                f"Scratch:\n"
            )

        elif strategy_id == 9:
            # Math-First Format: equation first, then explanation
            prompt = (
                f"{_SYSTEM_HEADER}"
                f"Question: {problem}\n\n"
                f"First, write the key equation(s), then explain:\n"
                f"Equation: "
            )

        # ── GROUP 4: CONTEXT-BASED ─────────────
        elif strategy_id == 10:
            # Problem Category Context
            hint = _fmt_category_hint(category_info) if category_info else ""
            prompt = (
                f"{_SYSTEM_HEADER}"
                f"{hint}"
                f"Question: {problem}\n"
                f"Let's think step by step.\n"
            )

        elif strategy_id == 11:
            # Worked Example + Explanation (1 detailed exemplar)
            worked = ""
            if exemplars:
                best = exemplars[0]
                worked = (
                    f"Worked Example:\n"
                    f"Question: {best['question']}\n"
                    f"Detailed Solution:\n{best['trace']}\n\n"
                    f"Notice how each step is clearly explained above.\n\n"
                )
            prompt = (
                f"{_SYSTEM_HEADER}"
                f"{worked}"
                f"Now apply the same detailed approach:\n"
                f"Question: {problem}\n"
                f"Detailed Solution:\n"
            )

        elif strategy_id == 12:
            # Reversed: show answer first, then question (expected to be worse)
            ans_hint = ""
            if exemplars:
                ex = exemplars[0]
                ex_ans = extract_final_answer(ex.get('trace', ''))
                ans_hint = (
                    f"Example — Answer: {ex_ans}\n"
                    f"Question that led to this: {ex['question']}\n\n"
                )
            prompt = (
                f"{_SYSTEM_HEADER}"
                f"{ans_hint}"
                f"Question: {problem}\n"
                f"Answer:"
            )

        # ── GROUP 5: HYBRID ────────────────────
        elif strategy_id == 13:
            # CoT + Few-Shot
            prompt = (
                f"{_SYSTEM_HEADER}"
                f"Here are some examples with step-by-step reasoning:\n\n"
                f"{_fmt_exemplars(exemplars, 3)}"
                f"Now solve this step by step:\n"
                f"Question: {problem}\n"
                f"Let's think step by step.\n"
            )

        elif strategy_id == 14:
            # CoT + Self-Consistency (3 generations)
            prompt = (
                f"{_SYSTEM_HEADER}"
                f"{_fmt_exemplars(exemplars, 2)}"
                f"Question: {problem}\n"
                f"Let's think step by step.\n"
            )
            gen_kwargs['num_return_sequences'] = 3
            gen_kwargs['temperature'] = 0.9

        elif strategy_id == 15:
            # Full Pipeline: CoT + exemplars + formatting + category context
            hint = _fmt_category_hint(category_info) if category_info else ""
            prompt = (
                f"{_SYSTEM_HEADER}"
                f"{hint}"
                f"Here are similar solved problems:\n\n"
                f"{_fmt_exemplars(exemplars, 3)}"
                f"Now solve this problem using numbered steps and "
                f"end with #### <answer>:\n\n"
                f"Question: {problem}\n\n"
                f"Step 1:"
            )

        else:
            raise ValueError(f"Unknown strategy_id: {strategy_id}")

        return prompt, gen_kwargs

    # ── public API ────────────────────────────
    def generate(
        self,
        problem: str,
        strategy_id: int,
        exemplars: List[Dict] = None,
        category_info: Optional[Dict] = None,
        max_new_tokens: int = 512,
    ) -> Dict:
        """
        Generate a reasoning trace for `problem` using `strategy_id`.

        For self-consistency strategies (4, 14), runs majority vote internally
        and returns the winning trace along with all candidates.

        Returns:
            {
                'trace'         : str   — final (or winning) trace,
                'answer'        : str   — extracted final answer,
                'strategy_id'   : int,
                'strategy_name' : str,
                'num_tokens'    : int   — tokens in winning trace,
                'candidates'    : list  — all traces (only for SC strategies),
            }
        """
        exemplars = exemplars or []
        strategy_name = self.STRATEGY_NAMES[strategy_id]

        prompt, gen_overrides = self._build_prompt(
            strategy_id, problem, exemplars, category_info
        )

        # Merge defaults with strategy overrides
        temperature         = gen_overrides.get('temperature', 0.7)
        num_return_sequences = gen_overrides.get('num_return_sequences', 1)

        traces = self._generate_text(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
        )

        # For self-consistency: pick majority-vote answer's trace
        if num_return_sequences > 1:
            answers   = [extract_final_answer(t) for t in traces]
            best_ans  = majority_vote(answers)
            # Pick the trace whose answer matches the majority
            best_trace = next(
                (t for t, a in zip(traces, answers) if a == best_ans),
                traces[0]
            )
        else:
            best_trace = traces[0]
            answers    = [extract_final_answer(best_trace)]
            best_ans   = answers[0]

        return {
            'trace'         : best_trace,
            'answer'        : best_ans,
            'strategy_id'   : strategy_id,
            'strategy_name' : strategy_name,
            'num_tokens'    : len(self.tokenizer.encode(best_trace)),
            'candidates'    : traces if num_return_sequences > 1 else [],
        }