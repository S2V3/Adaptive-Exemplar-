#!/usr/bin/env python3
"""
GENERATION MODULE — 16 Strategies
===================================
Each strategy is self-contained with its own prompt template.
STRATEGY_EXEMPLAR_NEEDS mirrors exemplar_selector.py (source of truth here;
selector imports this dict to avoid circular imports).

generate() accepts:
  - manual_exemplars : pre-selected seed exemplars for this strategy
  - pool_exemplars   : accumulated good pool (for dynamic KNN retrieval)
  - category_info    : from categorizer
  - token_logprobs   : if True, returns logprobs list alongside trace
"""

import re
try:
    import torch
    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    torch = None
    _TORCH_AVAILABLE = False
from collections import Counter
from typing import List, Dict, Optional, Tuple


# ── Strategy metadata ─────────────────────────────────────────────────────────

STRATEGY_NAMES: Dict[int, str] = {
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

# How many manual exemplars each strategy needs (from ExemplarSelector)
STRATEGY_EXEMPLAR_NEEDS: Dict[int, int] = {
    0: 0, 1: 0, 2: 5, 3: 8, 4: 0, 5: 0, 6: 0,
    7: 3, 8: 3, 9: 3, 10: 8, 11: 1, 12: 1,
    13: 5, 14: 3, 15: 8,
}

# Which strategies use majority vote (multiple generation sequences)
SELF_CONSISTENCY_STRATEGIES = {4, 14}
SELF_CONSISTENCY_N = {4: 5, 14: 3}


# ── Shared answer parser ──────────────────────────────────────────────────────

def extract_final_answer(text: str) -> Optional[str]:
    m = re.search(r'####\s*([\d,]+(?:\.\d+)?)', text)
    if m:
        return m.group(1).replace(',', '').strip()
    m = re.search(r'(?:the\s+)?answer\s*(?:is|=|:)\s*([\d,]+(?:\.\d+)?)',
                  text, re.IGNORECASE)
    if m:
        return m.group(1).replace(',', '').strip()
    nums = re.findall(r'[\d,]+(?:\.\d+)?', text)
    return nums[-1].replace(',', '').strip() if nums else None


def majority_vote(answers: List[Optional[str]]) -> Optional[str]:
    valid = [a for a in answers if a is not None]
    return Counter(valid).most_common(1)[0][0] if valid else None


# ── Prompt formatting helpers ─────────────────────────────────────────────────

_HEADER = (
    "You are an expert math tutor. Solve the following grade-school math "
    "problem carefully and end your answer with #### <number>.\n\n"
)

def _fmt_exemplars(exemplars: List[Dict], n: int) -> str:
    out = ""
    for i, ex in enumerate(exemplars[:n], 1):
        out += f"Example {i}:\nQuestion: {ex['question']}\nSolution: {ex['trace']}\n\n"
    return out

def _fmt_category_hint(cat_info: Dict) -> str:
    if not cat_info:
        return ""
    return (
        f"[Problem type: {cat_info.get('category_type','?').upper()} | "
        f"Complexity: {cat_info.get('complexity','?')}]\n\n"
    )


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(
    sid: int,
    problem: str,
    manual_exemplars: List[Dict],
    pool_exemplars: List[Dict],
    cat_info: Optional[Dict],
    knn_fn=None,                     # callable(query, pool, k, cat) → List[Dict]
) -> Tuple[str, Dict]:
    """
    Returns (prompt_string, generation_kwargs_overrides).
    """
    gen_kw = {}

    # Dynamic exemplars: use KNN at generation time
    def _dyn(k=3):
        if knn_fn and pool_exemplars:
            return knn_fn(problem, pool_exemplars, k,
                          cat_info.get('category_type') if cat_info else None)
        return manual_exemplars[:k]

    # ── GROUP 1: PROMPT-BASED ────────────────────────────────────────────────
    if sid == 0:
        prompt = f"{_HEADER}Question: {problem}\nAnswer:"

    elif sid == 1:
        prompt = (f"{_HEADER}Question: {problem}\n"
                  f"Let's think step by step.\n")

    elif sid == 2:
        prompt = (f"{_HEADER}{_fmt_exemplars(manual_exemplars, 5)}"
                  f"Question: {problem}\nSolution:")

    elif sid == 3:
        exemplars = _dyn(k=3)
        prompt = (f"{_HEADER}Here are similar problems:\n\n"
                  f"{_fmt_exemplars(exemplars, 3)}"
                  f"Now solve:\nQuestion: {problem}\nSolution:")

    # ── GROUP 2: SAMPLING-BASED ──────────────────────────────────────────────
    elif sid == 4:
        prompt = (f"{_HEADER}Question: {problem}\n"
                  f"Let's think step by step.\n")
        gen_kw['num_return_sequences'] = 5
        gen_kw['temperature']          = 0.9

    elif sid == 5:
        prompt = (f"{_HEADER}Question: {problem}\n"
                  f"Let's think step by step.\n")
        gen_kw['temperature'] = 0.3

    elif sid == 6:
        prompt = (f"{_HEADER}Question: {problem}\n"
                  f"Let's think step by step.\n")
        gen_kw['temperature'] = 1.2

    # ── GROUP 3: FORMATTING-BASED ────────────────────────────────────────────
    elif sid == 7:
        ex_block = ""
        if manual_exemplars:
            ex = manual_exemplars[0]
            ex_block = (f"Example:\nQuestion: {ex['question']}\n"
                        f"Step 1: [first sub-problem]\n"
                        f"Step 2: [next computation]\n"
                        f"Solution: {ex.get('trace','')}\n\n")
        prompt = (f"{_HEADER}{ex_block}"
                  f"Use numbered steps.\nQuestion: {problem}\n\nStep 1:")

    elif sid == 8:
        prompt = (f"{_HEADER}Question: {problem}\n\n"
                  f"Scratch:\n")

    elif sid == 9:
        prompt = (f"{_HEADER}Question: {problem}\n\n"
                  f"Key equation(s): ")

    # ── GROUP 4: CONTEXT-BASED ───────────────────────────────────────────────
    elif sid == 10:
        hint = _fmt_category_hint(cat_info)
        prompt = (f"{_HEADER}{hint}"
                  f"Question: {problem}\n"
                  f"Let's think step by step.\n")

    elif sid == 11:
        best = manual_exemplars[0] if manual_exemplars else None
        worked = ""
        if best:
            worked = (f"Worked example (study this carefully):\n"
                      f"Question: {best['question']}\n"
                      f"Full solution:\n{best.get('trace','')}\n\n"
                      f"Apply the same rigorous approach:\n\n")
        prompt = f"{_HEADER}{worked}Question: {problem}\nFull solution:\n"

    elif sid == 12:
        ex_ans = ""
        if manual_exemplars:
            ex     = manual_exemplars[0]
            ex_ans_val = extract_final_answer(ex.get('trace', ''))
            ex_ans = (f"[Example — the answer is {ex_ans_val}, "
                      f"question: {ex['question']}]\n\n")
        prompt = f"{_HEADER}{ex_ans}Question: {problem}\nAnswer:"

    # ── GROUP 5: HYBRID ──────────────────────────────────────────────────────
    elif sid == 13:
        exemplars = _dyn(k=3)
        prompt = (f"{_HEADER}Examples with step-by-step reasoning:\n\n"
                  f"{_fmt_exemplars(exemplars, 3)}"
                  f"Now solve step by step:\n"
                  f"Question: {problem}\nLet's think step by step.\n")

    elif sid == 14:
        exemplars = _dyn(k=2)
        prompt = (f"{_HEADER}{_fmt_exemplars(exemplars, 2)}"
                  f"Question: {problem}\nLet's think step by step.\n")
        gen_kw['num_return_sequences'] = 3
        gen_kw['temperature']          = 0.9

    elif sid == 15:
        hint      = _fmt_category_hint(cat_info)
        exemplars = _dyn(k=3)
        prompt = (f"{_HEADER}{hint}"
                  f"Similar solved problems:\n\n"
                  f"{_fmt_exemplars(exemplars, 3)}"
                  f"Solve with numbered steps, end with #### <answer>:\n\n"
                  f"Question: {problem}\n\nStep 1:")

    else:
        raise ValueError(f"Unknown strategy_id: {sid}")

    return prompt, gen_kw


# ── Main generator class ──────────────────────────────────────────────────────

class ExemplarGenerator:

    def __init__(self, model=None, tokenizer=None):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = (
            next(model.parameters()).device if model else
            ('cuda' if torch.cuda.is_available() else 'cpu')
        )

    def _run_model(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        num_return_sequences: int = 1,
        return_logprobs: bool = False,
    ) -> Tuple[List[str], Optional[List]]:
        """
        Run HF model.generate(). Returns (list_of_traces, logprobs_or_None).
        """
        enc        = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=1024
        ).to(self.device)
        prompt_len = enc['input_ids'].shape[1]

        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens      = max_new_tokens,
                temperature         = max(temperature, 1e-4),
                top_p               = 0.95,
                do_sample           = temperature > 0.01,
                num_return_sequences= num_return_sequences,
                pad_token_id        = self.tokenizer.eos_token_id,
                output_scores       = return_logprobs,
                return_dict_in_generate = return_logprobs,
            )

        if return_logprobs:
            sequences = out.sequences
            scores    = out.scores   # tuple of (vocab_size,) tensors
        else:
            sequences = out if not return_logprobs else out.sequences
            scores    = None

        traces = []
        for seq in sequences:
            new_toks = seq[prompt_len:]
            traces.append(
                self.tokenizer.decode(new_toks, skip_special_tokens=True).strip()
            )

        # Build logprobs list: [(token_str, log_prob), ...]
        logprobs_out = None
        if return_logprobs and scores is not None:
            # Use first sequence only for logprob scoring
            logprobs_out = []
            generated_ids = sequences[0][prompt_len:]
            for step_idx, (tok_id, score_tensor) in enumerate(
                zip(generated_ids, scores)
            ):
                log_probs = torch.log_softmax(score_tensor[0], dim=-1)
                lp        = log_probs[tok_id].item()
                tok_str   = self.tokenizer.decode([tok_id])
                logprobs_out.append((tok_str, lp))

        return traces, logprobs_out

    def generate(
        self,
        problem          : str,
        strategy_id      : int,
        manual_exemplars : List[Dict] = None,
        pool_exemplars   : List[Dict] = None,
        category_info    : Optional[Dict] = None,
        max_new_tokens   : int = 512,
        return_logprobs  : bool = True,
        knn_fn           = None,
    ) -> Dict:
        """
        Generate reasoning trace using strategy `strategy_id`.

        Args:
            problem          : GSM8K question string
            strategy_id      : 0-15
            manual_exemplars : pre-selected seed exemplars for this strategy
            pool_exemplars   : accumulated good pool for KNN retrieval (strategy 3,13-15)
            category_info    : from QuestionCategorizer.categorize()
            max_new_tokens   : generation length
            return_logprobs  : whether to return token log-probs (for K metric)
            knn_fn           : ExemplarSelector.get_knn_exemplars (optional)

        Returns:
            {
                'trace'          : str,
                'answer'         : str | None,
                'strategy_id'    : int,
                'strategy_name'  : str,
                'num_tokens'     : int,
                'candidates'     : list  (SC strategies only),
                'token_logprobs' : list  (if return_logprobs=True),
            }
        """
        manual_exemplars = manual_exemplars or []
        pool_exemplars   = pool_exemplars   or []

        prompt, gen_kw = _build_prompt(
            strategy_id, problem,
            manual_exemplars, pool_exemplars,
            category_info, knn_fn
        )

        temperature          = gen_kw.get('temperature', 0.7)
        num_return_sequences = gen_kw.get('num_return_sequences', 1)

        traces, logprobs = self._run_model(
            prompt,
            temperature          = temperature,
            max_new_tokens       = max_new_tokens,
            num_return_sequences = num_return_sequences,
            return_logprobs      = return_logprobs and (num_return_sequences == 1),
        )

        # Self-consistency: majority vote across candidates
        if num_return_sequences > 1:
            answers    = [extract_final_answer(t) for t in traces]
            best_ans   = majority_vote(answers)
            best_trace = next(
                (t for t, a in zip(traces, answers) if a == best_ans),
                traces[0]
            )
            logprobs = None   # not computed for multi-sequence
        else:
            best_trace = traces[0]
            best_ans   = extract_final_answer(best_trace)

        return {
            'trace'         : best_trace,
            'answer'        : best_ans,
            'strategy_id'   : strategy_id,
            'strategy_name' : STRATEGY_NAMES[strategy_id],
            'num_tokens'    : len(self.tokenizer.encode(best_trace))
                             if self.tokenizer else len(best_trace.split()),
            'candidates'    : traces if num_return_sequences > 1 else [],
            'token_logprobs': logprobs,
        }