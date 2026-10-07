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

"""Pins the pt-PT failure modes of the English Search-R1 EM scorer."""

import unicodedata

import pytest

from verl.utils.reward_score import search_r1_like_qa_em as en
from verl.utils.reward_score import search_r1_like_qa_em_ptpt as pt


def _answer(text):
    return f"<think>...</think><answer>{text}</answer>"


# (prediction, gold) pairs that are the same answer in pt-PT.
EQUIVALENT = [
    ("Marquês de Pombal", "o Marquês de Pombal"),   # masculine article
    ("Guerra Peninsular", "a Guerra Peninsular"),   # feminine article
    ("Descobrimentos", "os Descobrimentos"),        # plural article
    ("Sao Tome", "São Tomé"),                       # unaccented model output
    ("Lisboa-Porto", "Lisboa Porto"),               # hyphen vs space
    ("D. Afonso Henriques", "D Afonso Henriques"),  # punctuation
    ("25 de Abril", "25 de abril"),                 # case
    ("  Coimbra ", "Coimbra"),                      # whitespace
    ("6ª", "6.ª"),                                  # ordinal indicator (NFKC would make it "6a")
    ("1º", "1.º"),                                  # masculine ordinal
    ("3646", "3 646"),                              # digit grouping with a space
    ("30295", "30.295"),                            # digit grouping with a dot
    ("3646", "3 646"),                         # no-break space
    ("1000000", "1 000 000"),                       # several groups
    ("1143,6", "1 143,6"),                          # grouping before a decimal comma
]


@pytest.mark.parametrize("pred,gold", EQUIVALENT)
def test_ptpt_scorer_accepts_equivalent_answers(pred, gold):
    assert pt.compute_score(_answer(pred), {"target": [gold]}) == 1.0


def test_english_scorer_rejects_some_of_them():
    """Guards the premise: at least one pair is a false negative upstream."""
    false_negatives = [
        (p, g) for p, g in EQUIVALENT if en.compute_score(_answer(p), {"target": [g]}) != 1.0
    ]
    assert false_negatives, "English scorer no longer fails on pt-PT; this module may be redundant"


def test_nfc_nfd_forms_agree():
    nfc = unicodedata.normalize("NFC", "São Tomé")
    nfd = unicodedata.normalize("NFD", "São Tomé")
    assert nfc != nfd  # different bytes, identical rendering
    assert pt.compute_score(_answer(nfd), {"target": [nfc]}) == 1.0


def test_distinct_answers_still_score_zero():
    assert pt.compute_score(_answer("Porto"), {"target": ["Lisboa"]}) == 0.0
    assert pt.compute_score(_answer("1143"), {"target": ["1139"]}) == 0.0


@pytest.mark.parametrize(
    "pred,gold",
    [
        ("15", "1,5"),       # decimal comma is not a thousands separator
        ("3646", "364 6"),   # a group must have exactly three digits
        ("12345", "12 3456"),
        ("1985 123", "1985123"),
    ],
)
def test_digit_grouping_only_joins_thousands_groups(pred, gold):
    assert pt.compute_score(_answer(pred), {"target": [gold]}) == 0.0


def test_missing_answer_tag_scores_zero():
    assert pt.compute_score("nao sei", {"target": ["Lisboa"]}) == 0.0


def test_empty_prediction_scores_zero():
    assert pt.compute_score(_answer(""), {"target": ["Lisboa"]}) == 0.0
    assert pt.compute_score(_answer("o"), {"target": ["Lisboa"]}) == 0.0


def test_ground_truth_accepts_list_and_str():
    assert pt.compute_score(_answer("Lisboa"), ["Lisboa"]) == 1.0
    assert pt.compute_score(_answer("Lisboa"), "Lisboa") == 1.0


def test_alias_list_any_match():
    gold = {"target": ["Luís de Camões", "Camões"]}
    assert pt.compute_score(_answer("Camoes"), gold) == 1.0


def test_answer_tag_spam_is_penalised():
    spam = "<answer>Lisboa</answer>" * 11
    assert pt.compute_score(spam, {"target": ["Lisboa"]}) == 0.25


def test_subem_matches_within_sentence():
    sol = _answer("A resposta e Lisboa, a capital.")
    assert pt.compute_score_subem(sol, {"target": ["Lisboa"]}) == 1.0


def test_subem_empty_gold_does_not_match_everything():
    assert pt.compute_score_subem(_answer("Lisboa"), {"target": ["o"]}) == 0.0
