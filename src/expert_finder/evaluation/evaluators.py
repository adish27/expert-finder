"""Scorers for one agent answer.

Each function has the signature LangSmith expects - (inputs, outputs, reference_outputs) - so the
same objects produce scores in the LangSmith UI and in the terminal report; LangSmith binds
those arguments by name, so each scorer declares only the ones it reads.

Most scoring here is deterministic: whether a recommended company is actually good is a lookup
against ground truth, not a matter of opinion, and a deterministic scorer cannot drift between
runs. Only two judgements genuinely need reading the prose - did the answer raise a trust problem,
and did it call a price high or fair - and those go to one LLM judge that returns several scores
from a single call.

An evaluator that does not apply to a question returns a None score rather than nothing, which
LangSmith records as "not applicable" instead of logging an error.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from expert_finder.evaluation.ground_truth import PROBLEM_TEXT, GroundTruth

# Questions where the answer is a recommendation, so who it names can be scored.
RECOMMENDING_KINDS = frozenset({"recommend", "neighborhood", "expert"})
VERDICT_SCORE = {"good": 1.0, "mixed": 0.5, "poor": 0.0}
PRICE_SCALE = ["low", "typical", "high", "very high"]

NOT_APPLICABLE = "not applicable to this kind of question"


def _skip(key: str) -> dict:
    return {"key": key, "score": None, "comment": NOT_APPLICABLE}


class Judgement(BaseModel):
    """What only reading the answer can establish."""

    problems_mentioned: list[str] = Field(
        default_factory=list,
        description="Keys of the listed problems the answer actually raises. Empty if none.")
    recommends_the_business: bool = Field(
        default=False, description="True if the answer encourages hiring the company asked about.")
    price_verdict: str = Field(
        default="not stated",
        description="How the answer characterises the quoted price: low, typical, high, "
                    "very high, or 'not stated'.")


JUDGE_PROMPT = """You are grading an assistant's answer to a homeowner. Judge only what the answer \
says. Do not use outside knowledge, and do not reward an answer for being cautious in general.

Homeowner's question:
{question}

Assistant's answer:
{answer}

{task}

Be strict: a problem counts as raised only if the answer actually tells the homeowner about it."""

PROBLEM_TASK = """Which of these problems does the answer raise? Reply with the keys of the ones \
it raises, and say whether the answer encourages hiring this company.

{problems}"""

PRICE_TASK = """How does the answer characterise the quoted price: low, typical, high, very high, \
or not stated?"""


def build_evaluators(gt: GroundTruth, judge_model: Any | None = None) -> list[Callable]:
    """Every scorer, bound to this dataset's ground truth and (optionally) a judge model."""

    def recommendation_quality(inputs, outputs) -> dict:
        """The ground-truth standing of the companies the answer actually put forward.

        Scored over what was recommended rather than as recall against every good company in
        town: a homeowner wants two or three solid names, not all eleven.
        """
        if inputs["kind"] not in RECOMMENDING_KINDS:
            return _skip("recommendation_quality")
        cited = [b for b in outputs["cited_businesses"] if b in gt.businesses]
        if not cited:
            return {"key": "recommendation_quality", "score": 0.0,
                    "comment": "the answer named no business with a citable id"}
        verdicts = [gt.businesses[b].verdict for b in cited]
        score = sum(VERDICT_SCORE[v] for v in verdicts) / len(verdicts)
        return {"key": "recommendation_quality", "score": round(score, 3),
                "comment": ", ".join(f"{b} ({v})" for b, v in zip(cited, verdicts))}

    def no_poor_business_recommended(inputs, outputs) -> dict:
        """Binary, because recommending one bad contractor is the failure that matters."""
        if inputs["kind"] not in RECOMMENDING_KINDS:
            return _skip("no_poor_business_recommended")
        bad = [b for b in outputs["cited_businesses"]
               if b in gt.businesses and gt.businesses[b].verdict == "poor"]
        return {"key": "no_poor_business_recommended", "score": 0.0 if bad else 1.0,
                "comment": f"recommended {', '.join(bad)}" if bad else "none of the poor ones"}

    def local_word_of_mouth_precision(inputs, outputs, reference_outputs) -> dict:
        """For neighbourhood questions, were the names the ones that neighbourhood really uses?"""
        if inputs["kind"] != "neighborhood":
            return _skip("local_word_of_mouth_precision")
        expected = set(reference_outputs["expected"])
        cited = outputs["cited_businesses"]
        if not cited:
            return {"key": "local_word_of_mouth_precision", "score": 0.0, "comment": "no business named"}
        hits = [b for b in cited if b in expected]
        return {"key": "local_word_of_mouth_precision", "score": round(len(hits) / len(cited), 3),
                "comment": f"{len(hits)} of {len(cited)} named are locally recommended"}

    def names_the_expected_person(inputs, outputs, reference_outputs) -> dict:
        """The one case with a single right answer: the technician people ask for by name."""
        if inputs["kind"] != "expert":
            return _skip("names_the_expected_person")
        answer = outputs["answer"].lower()
        names = reference_outputs["expert_names"]
        found = [n for n in names if n.split()[0].lower() in answer or n.split()[-1].lower() in answer]
        return {"key": "names_the_expected_person", "score": 1.0 if found else 0.0,
                "comment": f"expected {', '.join(names)}"}

    def citation_support(outputs) -> dict:
        """Share of the answer's citations that trace back to something a tool returned."""
        cited, unsupported = len(outputs["cited_refs"]), len(outputs["unsupported_refs"])
        if not cited and not outputs["evidence_items"]:
            # The price and quote tools return statistics, not records with ids. Nothing was
            # citable, so citing nothing is right, and scoring it zero would punish the answer
            # for the shape of the question.
            return {"key": "citation_support", "score": None,
                    "comment": "no tool returned anything citable"}
        if not cited:
            return {"key": "citation_support", "score": 0.0, "comment": "the answer cites nothing"}
        return {"key": "citation_support", "score": round((cited - unsupported) / cited, 3),
                "comment": f"{unsupported} of {cited} citations match no retrieved evidence"}

    def grounded(outputs) -> dict:
        """The agent's own verification: no invented citations, no invented phone numbers."""
        return {"key": "grounded", "score": float(outputs["grounded"]),
                "comment": "; ".join(outputs["problems"]) or "clean"}

    def latency_ms(outputs) -> dict:
        return {"key": "latency_ms", "score": outputs["latency_ms"]}

    def cost_usd(outputs) -> dict:
        return {"key": "cost_usd", "score": outputs["cost_usd"]}

    evaluators = [recommendation_quality, no_poor_business_recommended, local_word_of_mouth_precision,
                  names_the_expected_person, citation_support, grounded, latency_ms, cost_usd]
    if judge_model is not None:
        evaluators.append(_build_judge(judge_model))
    return evaluators


def _build_judge(model: Any) -> Callable:
    """One LLM call per answer, returning every score that needs the prose read."""
    judge = model.with_structured_output(Judgement)

    def answer_judge(inputs, outputs, reference_outputs) -> list[dict]:
        kind = inputs["kind"]
        if kind == "trust":
            expected = reference_outputs["problems"]
            task = PROBLEM_TASK.format(problems="\n".join(f"- {key}: {PROBLEM_TEXT[key]}"
                                                          for key in expected))
            keys = ["trust_problems_surfaced", "appropriate_caution"]
        elif kind == "quote":
            task = PRICE_TASK
            keys = ["quote_verdict"]
        else:
            return [_skip("trust_problems_surfaced"), _skip("appropriate_caution"),
                    _skip("quote_verdict")]

        try:
            verdict = judge.invoke(JUDGE_PROMPT.format(question=inputs["question"],
                                                       answer=outputs["answer"], task=task))
        except Exception as e:
            # A judge failure is a missing measurement, not a failing answer, so it must not be
            # recorded as a zero.
            return [{"key": key, "score": None, "comment": f"judge failed: {type(e).__name__}"}
                    for key in keys]

        if kind == "trust":
            expected = reference_outputs["problems"]
            raised = [k for k in verdict.problems_mentioned if k in expected]
            return [
                {"key": "trust_problems_surfaced",
                 "score": round(len(raised) / len(expected), 3) if expected else None,
                 "comment": f"raised {raised or 'none'} of {expected}"},
                {"key": "appropriate_caution",
                 "score": 0.0 if verdict.recommends_the_business else 1.0,
                 "comment": "recommends a company it should warn about"
                            if verdict.recommends_the_business else "did not endorse them"},
            ]
        return [{"key": "quote_verdict", "score": _price_agreement(verdict.price_verdict,
                                                                   reference_outputs["verdict"]),
                 "comment": f"said {verdict.price_verdict!r}, expected {reference_outputs['verdict']!r}"}]

    return answer_judge


def _price_agreement(said: str, expected: str) -> float:
    """Full credit for the right band, half for the neighbouring one, nothing for the rest.

    "high" when the truth is "very high" is a smaller error than "typical", and a scale that
    cannot tell those apart would hide most of what changes between runs.
    """
    said = (said or "").strip().lower()
    if said == expected:
        return 1.0
    if said in PRICE_SCALE and abs(PRICE_SCALE.index(said) - PRICE_SCALE.index(expected)) == 1:
        return 0.5
    return 0.0
