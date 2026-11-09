"""Run the question set through one or both retrieval modes and score the answers.

Both modes run against the same questions, the same model and the same evaluators, so the
difference between two experiments is the difference the graph makes.

With a LangSmith key the question set is uploaded as a dataset and each mode becomes a traced
experiment, so individual runs can be opened and compared. Without one, `upload_results=False`
runs the identical code path locally and only the terminal report comes back.
"""

from __future__ import annotations

import datetime
import re
import uuid
from pathlib import Path
from typing import Any

from langsmith import Client, evaluate
from langsmith.schemas import Example

from expert_finder.agent import build_agent, run_agent
from expert_finder.app import ExpertFinder, Mode
from expert_finder.dataset import Dataset
from expert_finder.evaluation.evaluators import build_evaluators
from expert_finder.evaluation.ground_truth import GroundTruth
from expert_finder.evaluation.questions import EvalQuestion, build_questions

DATASET_NAME = "home-services-expert-finder"
BUSINESS_REF = re.compile(r"^biz_\d+$")

# Metrics where a lower number is the better one, so the report does not read backwards.
LOWER_IS_BETTER = {"latency_ms", "cost_usd"}


def _cost_usd(app: ExpertFinder, stats: dict) -> float:
    s = app.settings
    return round(stats.get("input_tokens", 0) / 1e6 * s.price_per_mtok_input
                 + stats.get("output_tokens", 0) / 1e6 * s.price_per_mtok_output, 6)


def _target(app: ExpertFinder, mode: Mode):
    """The system under test: one agent, one fresh conversation per question."""
    agent, recorder = build_agent(app, mode)

    def answer(inputs: dict) -> dict:
        result = run_agent(app, inputs["question"], mode=mode, agent=agent, recorder=recorder,
                           thread_id=f"eval-{mode}-{inputs['id']}")
        return {
            "answer": result.answer,
            "cited_refs": result.cited_refs,
            "unsupported_refs": result.unsupported_refs,
            # The prompt requires every recommended company to carry its [biz_...] id, which makes
            # "who did it recommend" a lookup rather than a guess at names in prose.
            "cited_businesses": [r for r in result.cited_refs if BUSINESS_REF.match(r)],
            "grounded": result.grounded,
            "problems": result.problems,
            "tool_calls": [c["name"] for c in result.tool_calls],
            "latency_ms": result.stats["latency_ms"],
            "cost_usd": _cost_usd(app, result.stats),
            "evidence_items": result.stats.get("evidence_items", 0),
            "graph_items": result.stats.get("graph_items", 0),
        }

    return answer


def _local_examples(questions: list[EvalQuestion]) -> list[Example]:
    """Examples built in memory, for a run that does not upload."""
    dataset_id = uuid.uuid4()
    now = datetime.datetime.now(datetime.UTC)
    out = []
    for q in questions:
        inputs, outputs = q.as_example()
        out.append(Example(id=uuid.uuid4(), dataset_id=dataset_id, created_at=now,
                           inputs=inputs, outputs=outputs))
    return out


def _select(examples: list[Example], questions: list[EvalQuestion]) -> list[Example]:
    """Just the examples for these questions, in the question set's own order.

    The experiment has to run the questions that were asked for. Handing `evaluate` a dataset by
    name would run every example the dataset has ever held, so --kind and --limit would be
    silently ignored and two runs over different slices would look comparable.
    """
    order = {q.id: i for i, q in enumerate(questions)}
    chosen = [e for e in examples if e.inputs.get("id") in order]
    missing = order.keys() - {e.inputs["id"] for e in chosen}
    if missing:
        raise RuntimeError(f"dataset is missing examples for {sorted(missing)}")
    return sorted(chosen, key=lambda e: order[e.inputs["id"]])


def _upload_dataset(client: Client, questions: list[EvalQuestion]) -> list[Example]:
    """Create the dataset if it is missing, add any question it does not have yet, and return
    the examples for this run.

    Question ids are deterministic, so this is safe to re-run: an existing dataset keeps its
    history and only genuinely new questions are added.
    """
    if not client.has_dataset(dataset_name=DATASET_NAME):
        client.create_dataset(DATASET_NAME,
                              description="Home services expert finder: questions generated from "
                                          "the dataset's hidden ground truth.")
        existing: set[str] = set()
    else:
        existing = {e.inputs.get("id") for e in client.list_examples(dataset_name=DATASET_NAME)}
    new = [q for q in questions if q.id not in existing]
    if new:
        client.create_examples(dataset_name=DATASET_NAME, examples=[
            {"inputs": inputs, "outputs": outputs}
            for inputs, outputs in (q.as_example() for q in new)])
    return _select(list(client.list_examples(dataset_name=DATASET_NAME)), questions)


def _aggregate(rows: list[dict]) -> dict[str, float]:
    """Mean of every score across the questions it applied to.

    Scores recorded as None are skipped rather than counted as zero: an evaluator that does not
    apply to a question, or a judge call that failed, is a missing measurement.
    """
    totals: dict[str, list[float]] = {}
    for row in rows:
        for result in row["evaluation_results"]["results"]:
            if result.score is not None:
                totals.setdefault(result.key, []).append(float(result.score))
    return {key: round(sum(values) / len(values), 3) for key, values in sorted(totals.items())}


def run_mode(app: ExpertFinder, mode: Mode, gt: GroundTruth, data: Any,
             questions: list[EvalQuestion], judge: bool, upload: bool) -> dict:
    """One experiment: every question through one retrieval mode."""
    evaluators = build_evaluators(gt, judge_model=app.chat_model() if judge else None)
    results = evaluate(_target(app, mode), data=data, evaluators=evaluators,
                       experiment_prefix=f"expert-finder-{mode}",
                       metadata={"mode": mode, "questions": len(questions),
                                 "chat_model": app.settings.chat_model,
                                 "multi_query_rewrites": app.settings.multi_query_rewrites,
                                 "top_k": app.settings.top_k},
                       # Sequential: these runs share one agent and one recorder, and the point is
                       # a comparable measurement rather than a fast one.
                       max_concurrency=1, upload_results=upload)
    rows = list(results)
    scores = _aggregate(rows)
    per_question = [{"id": row["example"].inputs["id"],
                     "kind": row["example"].inputs["kind"],
                     "scores": {r.key: r.score for r in row["evaluation_results"]["results"]
                                if r.score is not None}}
                    for row in rows]
    return {"mode": mode, "questions": len(rows), "scores": scores, "per_question": per_question,
            "experiment": results.experiment_name if upload else None,
            "url": results.url if upload else None}


def run_evaluation(app: ExpertFinder, modes: tuple[Mode, ...] = ("vector", "graph"),
                   limit: int | None = None, judge: bool = True, upload: bool = True,
                   kinds: tuple[str, ...] | None = None) -> dict:
    """Score every mode on the same questions, plus the mode-independent retrieval metrics.

    Ground truth is loaded here and never handed to `app`, so it stays out of reach of the agent
    and its tools.
    """
    ds = Dataset.load(app.settings.data_dir)
    gt = GroundTruth.load(ds, app.settings.data_dir)
    questions = [q for q in build_questions(ds, gt) if kinds is None or q.kind in kinds][:limit]
    if not questions:
        raise ValueError(f"no questions of kind {kinds}")

    client = Client() if upload else None
    if client is not None:
        try:
            data: Any = _upload_dataset(client, questions)
        except Exception as e:
            print(f"LangSmith unavailable ({type(e).__name__}: {e}); running locally instead.")
            client, upload = None, False
    if client is None:
        data = _local_examples(questions)

    report = {"questions": [q.id for q in questions],
              "settings": {"chat_model": app.settings.chat_model, "top_k": app.settings.top_k,
                           "multi_query_rewrites": app.settings.multi_query_rewrites},
              "entity_resolution": mention_scores(ds, app.settings.data_dir),
              "modes": {}}
    for mode in modes:
        report["modes"][mode] = run_mode(app, mode, gt, data, questions, judge, upload)
    return report


def mention_scores(ds: Dataset, data_dir: Path) -> dict:
    """Entity-resolution accuracy: one number for the whole dataset, the same in both modes.

    Reuses the scoring behind `expert-finder mention-report` rather than restating it, because
    how well "Mike at the plumbing place" resolves is a property of the graph build, not of an
    answer to any one question.
    """
    from expert_finder.entity_resolution import EntityResolver, evaluate_mentions, evaluate_review_mentions

    resolver = EntityResolver(ds)
    gt_dir = data_dir / "ground_truth"
    social = evaluate_mentions(ds, resolver, gt_dir / "social_mentions.jsonl")
    reviews = evaluate_review_mentions(ds, resolver, gt_dir / "review_expert_mentions.jsonl")
    return {"post_to_business": {k: social["business_mentions"][k] for k in ("precision", "recall")},
            "post_to_expert": {k: social["expert_mentions"][k] for k in ("precision", "recall")},
            "review_to_expert": {k: reviews[k] for k in ("precision", "recall")}}


# --- reporting ----------------------------------------------------------------------------------


# Metrics worth naming per question when they come back short of perfect.
BINARY_METRICS = ("grounded", "no_poor_business_recommended", "appropriate_caution")


def _failures(per_question: list[dict]) -> list[str]:
    out = []
    for q in per_question:
        bad = [f"{k} {q['scores'][k]}" for k in BINARY_METRICS if q["scores"].get(k) == 0.0]
        bad += [f"{k} {q['scores'][k]}" for k in ("citation_support",)
                if q["scores"].get(k) is not None and q["scores"][k] < 1.0]
        if bad:
            out.append(f"{q['id']} -> " + ", ".join(bad))
    return out


def format_report(report: dict) -> str:
    """The comparison table, for the terminal."""
    modes = list(report["modes"])
    keys = sorted({k for m in modes for k in report["modes"][m]["scores"]})
    width = max(len(k) for k in keys) + 2

    lines = [f"{len(report['questions'])} questions | "
             + " | ".join(f"{k}={v}" for k, v in report["settings"].items()), ""]
    lines.append("metric".ljust(width) + "".join(m.rjust(12) for m in modes) + "   better")
    lines.append("-" * (width + 12 * len(modes) + 9))
    for key in keys:
        values = [report["modes"][m]["scores"].get(key) for m in modes]
        cells = "".join(("-" if v is None else f"{v:,.3f}").rjust(12) for v in values)
        known = [(m, v) for m, v in zip(modes, values) if v is not None]
        if len(known) < 2 or known[0][1] == known[1][1]:
            better = "same" if len(known) > 1 else ""
        else:
            pick = min(known, key=lambda kv: kv[1]) if key in LOWER_IS_BETTER else max(known, key=lambda kv: kv[1])
            better = pick[0]
        lines.append(key.ljust(width) + cells + "   " + better)

    # Name the questions that failed, because a mean of 0.933 does not say which answer broke.
    for mode in modes:
        for failure in _failures(report["modes"][mode]["per_question"]):
            lines.append(f"{mode}: {failure}")

    lines.append("")
    er = report["entity_resolution"]
    lines.append("entity resolution (same in both modes): "
                 + "; ".join(f"{name} precision {v['precision']} recall {v['recall']}"
                             for name, v in er.items()))
    for mode in modes:
        url = report["modes"][mode].get("url")
        if url:
            lines.append(f"{mode}: {url}")
    return "\n".join(lines)
