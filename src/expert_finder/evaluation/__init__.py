"""Scoring the two retrieval modes against the dataset's hidden ground truth.

`ground_truth` is the only module here (or anywhere) that reads `data/home_services/ground_truth/`,
and it is never reachable from the agent or its tools.
"""

from expert_finder.evaluation.ground_truth import GroundTruth
from expert_finder.evaluation.questions import EvalQuestion, build_questions
from expert_finder.evaluation.runner import format_report, run_evaluation

__all__ = ["EvalQuestion", "GroundTruth", "build_questions", "format_report", "run_evaluation"]
