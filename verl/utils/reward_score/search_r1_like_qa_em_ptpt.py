# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# European Portuguese (pt-PT) variant of search_r1_like_qa_em.

"""Exact-match scoring for pt-PT search/QA RLVR.

The upstream scorer in ``search_r1_like_qa_em`` normalizes for English and
produces false negatives on Portuguese answers. This module keeps the same
public interface but fixes five failure modes:

1. Article stripping. Upstream removes only ``a|an|the``. In pt-PT that strips
   the feminine article ``a`` while leaving ``o``, ``os``, ``as``, ``um``,
   ``uma``, so ``"o Marquês de Pombal"`` never matches ``"Marquês de Pombal"``.
2. Unicode form. Upstream does no Unicode normalization, so NFC ``ã`` (U+00E3)
   and NFD ``a``+U+0303 compare unequal while rendering identically in logs.
   Mixing corpus sources makes that collision likely.
3. Diacritics. ``"São Tomé"`` vs ``"Sao Tome"`` scores 0 upstream. Folding is
   on by default: it removes a large class of false negatives and costs very
   little discrimination between distinct Portuguese answer spans.
4. Ordinal indicators. NFKC turns ``ª``/``º`` into the letters ``a``/``o``, so
   ``"6ª"`` became ``"6a"`` and never matched ``"6.ª"``. They are removed before
   NFKC, like the other punctuation.
5. Digit grouping. ``"3646"``, ``"3 646"`` and ``"3.646"`` are the same number;
   separators between groups of three digits are dropped.

Deliberate deviation from upstream: punctuation is replaced with a space rather
than deleted, so ``"Lisboa-Porto"`` and ``"Lisboa Porto"`` agree. Upstream
deletes, which collapses the first to ``lisboaporto``. Set
``PUNCT_TO_SPACE = False`` to restore upstream behaviour if you need strict
comparability with published Search-R1 numbers.

All normalization is applied symmetrically to prediction and gold, so stripping
can only cause false positives (two genuinely different answers collapsing),
never false negatives.
"""

import random
import re
import string
import unicodedata

from .search_r1_like_qa_em import count_answer_tags, extract_solution

__all__ = [
    "normalize_answer",
    "em_check",
    "subem_check",
    "extract_solution",
    "count_answer_tags",
    "compute_score",
    "compute_score_subem",
]

# Replace punctuation with a space instead of deleting it. See module docstring.
PUNCT_TO_SPACE = True

# Definite and indefinite articles. English forms are kept so that borrowed or
# code-switched answers ("the Beatles") normalize the same way.
_ARTICLES_PT = ["o", "a", "os", "as", "um", "uma", "uns", "umas"]
_ARTICLES_EN = ["a", "an", "the"]
_ARTICLE_RE = re.compile(
    r"\b(?:" + "|".join(sorted(set(_ARTICLES_PT + _ARTICLES_EN), key=len, reverse=True)) + r")\b"
)

# string.punctuation misses the marks that actually show up in Portuguese text.
_EXTRA_PUNCT = "«»“”„‘’–—―…ºª§¡¿"
_PUNCT = set(string.punctuation) | set(_EXTRA_PUNCT)

# Ordinal indicators must go before NFKC, which maps them to the letters "a"/"o".
_ORDINAL_INDICATORS = {ord("ª"): " ", ord("º"): " "}

# A number written in groups: 1-3 leading digits, then groups of exactly three
# separated by a space or dot ("3 646", "30.295", "1 000 000"). "1985 123" is not
# one. NFKC has already turned no-break spaces into plain ones. The decimal comma
# is left alone ("1 143,6" -> "1143,6").
_DIGIT_GROUP_RE = re.compile(r"(?<!\d)\d{1,3}(?:[ .]\d{3})+(?!\d)")


def _strip_diacritics(text: str) -> str:
    """Fold accents: NFD, then drop combining marks."""
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if not unicodedata.combining(ch))


def normalize_answer(s, fold_diacritics: bool = True) -> str:
    """Normalize a pt-PT answer span for exact-match comparison.

    Args:
        s: the answer text.
        fold_diacritics: if True (default), remove accents so that unaccented
            model output still matches an accented gold answer.
    """
    if s is None:
        return ""

    text = str(s).translate(_ORDINAL_INDICATORS)

    # NFKC next: unifies compatibility forms (e.g. full-width chars, ligatures)
    # and settles NFC/NFD before anything else inspects the string.
    text = unicodedata.normalize("NFKC", text).lower()

    text = _DIGIT_GROUP_RE.sub(lambda m: re.sub(r"[ .]", "", m.group()), text)

    if fold_diacritics:
        text = _strip_diacritics(text)

    if PUNCT_TO_SPACE:
        text = "".join(" " if ch in _PUNCT else ch for ch in text)
    else:
        text = "".join(ch for ch in text if ch not in _PUNCT)

    text = _ARTICLE_RE.sub(" ", text)
    return " ".join(text.split())


def _golden_answers(ground_truth):
    """Accept {"target": [...]}, a bare list, or a bare string."""
    if isinstance(ground_truth, dict):
        golden = ground_truth.get("target", [])
    else:
        golden = ground_truth
    if golden is None:
        return []
    if isinstance(golden, str):
        return [golden]
    return list(golden)


def em_check(prediction, golden_answers) -> int:
    golden_answers = _golden_answers(golden_answers)
    normalized_prediction = normalize_answer(prediction)
    if not normalized_prediction:
        return 0
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return 1
    return 0


def subem_check(prediction, golden_answers) -> int:
    golden_answers = _golden_answers(golden_answers)
    normalized_prediction = normalize_answer(prediction)
    if not normalized_prediction:
        return 0
    for golden_answer in golden_answers:
        normalized_golden = normalize_answer(golden_answer)
        # Guard against an empty gold (e.g. gold was just an article) matching
        # every prediction via the substring test.
        if normalized_golden and normalized_golden in normalized_prediction:
            return 1
    return 0


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """Exact-match score for pt-PT QA. Mirrors search_r1_like_qa_em.compute_score."""
    answer = extract_solution(solution_str=solution_str)
    open_count, close_count = count_answer_tags(solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {_golden_answers(ground_truth)}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return format_score
    if em_check(answer, ground_truth):
        if open_count > 10 or close_count > 10:  # prevent spamming </answer>
            return score / 4
        return score
    return format_score


def compute_score_subem(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """Substring exact-match score for pt-PT QA."""
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {_golden_answers(ground_truth)}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return format_score
    if subem_check(answer, ground_truth):
        return score
    return format_score
