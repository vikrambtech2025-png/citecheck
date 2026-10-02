"""Evaluation: task suite, automatic graders, sweeps, and reporting."""

from citecheck.eval.graders import Grade, aggregate, grade
from citecheck.eval.harness import (
    Harness,
    SuiteReport,
    TaskOutcome,
    run_arch_sweep,
    run_model_sweep,
)
from citecheck.eval.tasks import ALL_TASKS, Task, get_tasks, suite_summary

__all__ = [
    "ALL_TASKS",
    "Grade",
    "Harness",
    "SuiteReport",
    "Task",
    "TaskOutcome",
    "aggregate",
    "get_tasks",
    "grade",
    "run_arch_sweep",
    "run_model_sweep",
    "suite_summary",
]