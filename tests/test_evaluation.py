"""The evaluation harness, offline.

Three things are checked here. First, that the ground-truth labels are coherent, which is a test
of the labels themselves rather than of the code that reads them. Second, that the question set is
deterministic, because an evaluation whose questions move cannot compare two runs. Third, that
every deterministic scorer returns what it claims to - the LLM judge is exercised separately and
marked integration, since it needs a model.
"""

from __future__ import annotations

import datetime
import inspect
import uuid

import pytest

from expert_finder.config import PROJECT_ROOT, get_settings
from expert_finder.dataset import Dataset
from expert_finder.evaluation.evaluators import _price_agreement, build_evaluators
from expert_finder.evaluation.ground_truth import (
    GOOD_ARCHETYPES,
    POOR_ARCHETYPES,
    PROBLEM_TEXT,
    GroundTruth,
)
from expert_finder.evaluation.questions import PER_KIND, build_questions
from expert_finder.evaluation.runner import _aggregate, format_report

NOW = datetime.datetime.now(datetime.UTC)


@pytest.fixture(scope="module")
def ds():
    return Dataset.load(get_settings().data_dir)


@pytest.fixture(scope="module")
def gt(ds):
    return GroundTruth.load(ds, get_settings().data_dir)


@pytest.fixture(scope="module")
def questions(ds, gt):
    return build_questions(ds, gt)


# --- are the labels coherent? -------------------------------------------------------------------


def test_the_verdict_and_the_job_outcomes_agree(gt):
    """The two judgements are derived independently, so comparing them validates both.

    `verdict` comes from the generator's archetype and `outcome_score` only from what happened on
    the jobs. If companies labelled good did not in fact do better work, one of the two is wrong.
    """
    def mean_score(verdict):
        scores = [t.outcome_score for t in gt.businesses.values() if t.verdict == verdict]
        return sum(scores) / len(scores)

    assert mean_score("good") > mean_score("poor") + 0.1


def test_companies_with_too_little_history_are_neither_good_nor_poor(gt):
    """A young company is not a bad one, so recommending it must not be scored as a mistake."""
    mixed = {t.archetype for t in gt.businesses.values() if t.verdict == "mixed"}
    assert mixed == {"new_and_eager"}
    assert not mixed & (GOOD_ARCHETYPES | POOR_ARCHETYPES)


def test_every_business_is_labelled_and_scored(ds, gt):
    assert set(gt.businesses) == {b["business_id"] for b in ds.businesses}
    for truth in gt.businesses.values():
        assert 0.0 <= truth.outcome_score <= 1.0
        assert truth.verdict in {"good", "mixed", "poor"}
        assert truth.jobs > 0
        assert set(truth.problems) <= set(PROBLEM_TEXT), truth.business_id


def test_the_companies_with_planted_fake_reviews_are_all_flagged(gt):
    for truth in gt.with_archetype("suspicious_reviews"):
        assert "fake_reviews" in truth.problems
    for truth in gt.with_archetype("declining"):
        assert "declining" in truth.problems


def test_ground_truth_is_not_reachable_from_the_agent():
    """The assistant must not be able to read the labels it is scored against."""
    allowed = {"evaluation", "entity_resolution.py", "cli.py", "dataset.py", "documents.py"}
    for path in (PROJECT_ROOT / "src" / "expert_finder").rglob("*.py"):
        if "ground_truth" not in path.read_text():
            continue
        where = path.relative_to(PROJECT_ROOT / "src" / "expert_finder")
        assert where.parts[0] in allowed, f"{where} refers to ground truth"


# --- is the question set usable? -----------------------------------------------------------------


def test_the_same_dataset_always_produces_the_same_questions(ds, gt):
    assert build_questions(ds, gt) == build_questions(ds, gt)


def test_question_ids_are_unique_and_every_kind_is_covered(questions):
    ids = [q.id for q in questions]
    assert len(ids) == len(set(ids))
    kinds = {q.kind for q in questions}
    assert kinds == {"trust", "recommend", "neighborhood", "expert", "quote"}
    assert len(questions) == sum(PER_KIND.values())


def test_a_price_question_cannot_be_answered_by_always_saying_it_is_fair(questions):
    verdicts = {q.reference["verdict"] for q in questions if q.kind == "quote"}
    assert verdicts == {"typical", "very high"}


def test_trust_questions_expect_problems_that_exist(gt, questions):
    for q in (q for q in questions if q.kind == "trust"):
        truth = gt.businesses[q.reference["business_id"]]
        assert q.reference["problems"], q.id
        assert list(truth.problems) == q.reference["problems"]
        assert truth.name in q.question


def test_recommendation_questions_have_both_a_right_and_a_wrong_answer(gt, questions):
    for q in (q for q in questions if q.kind == "recommend"):
        category, city = q.reference["category"], q.reference["city"]
        assert gt.by_verdict("good", category, city)
        assert gt.by_verdict("poor", category, city), "nothing to get wrong is nothing to measure"


# --- do the scorers score? -----------------------------------------------------------------------


def answer(**overrides) -> dict:
    out = {"answer": "Call them.", "cited_refs": ["R1"], "unsupported_refs": [],
           "cited_businesses": [], "grounded": True, "problems": [], "tool_calls": [],
           "latency_ms": 1200, "cost_usd": 0.004, "evidence_items": 5, "graph_items": 0}
    return {**out, **overrides}


def score(evaluators, name, inputs, outputs, reference) -> dict:
    """Run one named scorer, passing only the arguments it declares, as LangSmith does."""
    available = {"inputs": inputs, "outputs": outputs, "reference_outputs": reference}
    for fn in evaluators:
        if fn.__name__ != name:
            continue
        wanted = inspect.signature(fn).parameters
        result = fn(**{k: v for k, v in available.items() if k in wanted})
        return result if isinstance(result, dict) else result[0]
    raise AssertionError(f"no evaluator named {name}")


@pytest.fixture(scope="module")
def evaluators(gt):
    return build_evaluators(gt, judge_model=None)


@pytest.fixture(scope="module")
def good_and_poor(gt):
    return (gt.by_verdict("good")[0].business_id, gt.by_verdict("poor")[0].business_id)


def test_recommending_good_companies_scores_higher_than_recommending_bad_ones(evaluators, good_and_poor):
    good, poor = good_and_poor
    inputs = {"kind": "recommend", "question": "q", "id": "x"}
    only_good = score(evaluators, "recommendation_quality", inputs,
                      answer(cited_businesses=[good]), {})
    only_poor = score(evaluators, "recommendation_quality", inputs,
                      answer(cited_businesses=[poor]), {})
    assert only_good["score"] == 1.0 and only_poor["score"] == 0.0


def test_naming_no_business_at_all_is_a_failed_recommendation(evaluators):
    result = score(evaluators, "recommendation_quality", {"kind": "recommend", "question": "q", "id": "x"},
                   answer(cited_businesses=[]), {})
    assert result["score"] == 0.0


def test_one_bad_recommendation_fails_the_binary_check(evaluators, good_and_poor):
    good, poor = good_and_poor
    inputs = {"kind": "recommend", "question": "q", "id": "x"}
    assert score(evaluators, "no_poor_business_recommended", inputs,
                 answer(cited_businesses=[good, poor]), {})["score"] == 0.0
    assert score(evaluators, "no_poor_business_recommended", inputs,
                 answer(cited_businesses=[good]), {})["score"] == 1.0


def test_scorers_that_do_not_apply_report_no_score_rather_than_zero(evaluators, good_and_poor):
    """A zero would drag the average down for questions the metric was never about."""
    _, poor = good_and_poor
    inputs = {"kind": "trust", "question": "q", "id": "x"}
    for name in ("recommendation_quality", "no_poor_business_recommended",
                 "local_word_of_mouth_precision", "names_the_expected_person"):
        result = score(evaluators, name, inputs, answer(cited_businesses=[poor]), {})
        assert result["score"] is None, name


def test_local_precision_counts_only_the_companies_that_neighborhood_uses(evaluators):
    inputs = {"kind": "neighborhood", "question": "q", "id": "x"}
    result = score(evaluators, "local_word_of_mouth_precision", inputs,
                   answer(cited_businesses=["biz_0061", "biz_0099"]),
                   {"expected": ["biz_0061", "biz_0064"]})
    assert result["score"] == 0.5


def test_the_expected_technician_is_recognised_by_either_name(evaluators):
    inputs = {"kind": "expert", "question": "q", "id": "x"}
    reference = {"expert_names": ["Bob Sandoval"]}
    for text, expected in [("Ask for Sandoval when you call.", 1.0),
                           ("Ask for Bob.", 1.0),
                           ("Anyone on the team is fine.", 0.0)]:
        assert score(evaluators, "names_the_expected_person", inputs,
                     answer(answer=text), reference)["score"] == expected


def test_citation_support_falls_with_invented_references(evaluators):
    inputs = {"kind": "recommend", "question": "q", "id": "x"}
    assert score(evaluators, "citation_support", inputs,
                 answer(cited_refs=["R1", "R2"], unsupported_refs=["R2"]), {})["score"] == 0.5
    assert score(evaluators, "citation_support", inputs, answer(cited_refs=[]), {})["score"] == 0.0


def test_citing_nothing_is_not_a_failure_when_nothing_was_citable(evaluators):
    """A price answer comes from statistics, not from records with ids."""
    result = score(evaluators, "citation_support", {"kind": "quote", "question": "q", "id": "x"},
                   answer(cited_refs=[], evidence_items=0), {})
    assert result["score"] is None


def test_groundedness_comes_from_the_agents_own_check(evaluators):
    inputs = {"kind": "recommend", "question": "q", "id": "x"}
    assert score(evaluators, "grounded", inputs, answer(grounded=False, problems=["bad"]), {})["score"] == 0.0
    assert score(evaluators, "grounded", inputs, answer(), {})["score"] == 1.0


@pytest.mark.parametrize("said, expected, want", [
    ("very high", "very high", 1.0),
    ("high", "very high", 0.5),       # the neighbouring band is a smaller error
    ("typical", "very high", 0.0),
    ("not stated", "typical", 0.0),
    ("", "typical", 0.0),
])
def test_price_bands_give_partial_credit_to_a_near_miss(said, expected, want):
    assert _price_agreement(said, expected) == want


# --- aggregation and reporting -------------------------------------------------------------------


class FakeResult:
    def __init__(self, key, score):
        self.key, self.score = key, score


def rows(*scores):
    return [{"evaluation_results": {"results": [FakeResult(k, v) for k, v in s.items()]}}
            for s in scores]


def test_a_metric_that_did_not_apply_is_left_out_of_the_mean():
    aggregated = _aggregate(rows({"grounded": 1.0, "quote_verdict": None},
                                 {"grounded": 0.0, "quote_verdict": 1.0}))
    assert aggregated == {"grounded": 0.5, "quote_verdict": 1.0}


def test_the_report_says_which_mode_won_each_metric():
    report = {
        "questions": ["q1"],
        "settings": {"chat_model": "gpt-4.1-mini"},
        "entity_resolution": {"post_to_business": {"precision": 0.9, "recall": 0.8}},
        "modes": {
            "vector": {"scores": {"grounded": 0.5, "latency_ms": 9000.0},
                       "per_question": [{"id": "q1", "kind": "trust", "scores": {"grounded": 0.0}}]},
            "graph": {"scores": {"grounded": 0.9, "latency_ms": 12000.0},
                      "per_question": [{"id": "q1", "kind": "trust", "scores": {"grounded": 1.0}}]},
        },
    }
    text = format_report(report)
    lines = {line.split()[0]: line for line in text.splitlines() if line.strip()}
    assert lines["grounded"].endswith("graph")
    assert lines["latency_ms"].endswith("vector"), "lower latency is the better latency"
    # A mean of 0.5 does not say which answer broke, so the failing question is named.
    assert "vector: q1 -> grounded 0.0" in text
    assert "graph: q1" not in text, "the mode that passed has nothing to report"


def test_an_experiment_runs_only_the_questions_that_were_asked_for(questions):
    """Handing LangSmith a dataset by name would run every example it has ever held."""
    from langsmith.schemas import Example

    from expert_finder.evaluation.runner import _select

    chosen = [q for q in questions if q.kind == "quote"]
    stored = [Example(id=uuid.uuid4(), dataset_id=uuid.uuid4(), created_at=NOW,
                      inputs={"id": q.id, "kind": q.kind, "question": q.question},
                      outputs=q.reference)
              for q in reversed(questions)]          # stored in some other order

    selected = _select(stored, chosen)
    assert [e.inputs["id"] for e in selected] == [q.id for q in chosen]


def test_a_dataset_missing_a_question_is_an_error_not_a_quiet_short_run(questions):
    from expert_finder.evaluation.runner import _select

    with pytest.raises(RuntimeError, match="missing examples"):
        _select([], questions[:1])
