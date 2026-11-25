"""Streaming inference and label-free penalty calibration for Statistical Jump Models."""

from .decoder import (
    OnlineJumpDecoder,
    compute_loss_matrix,
    default_penalty_matrix,
    fit_jump_model,
)
from .diagnostics import (
    ComparisonRow,
    ConfirmationFilter,
    PenaltyPoint,
    characteristic_loss_scale,
    compare_carried_vs_rolling,
    evaluate_penalty_curve,
    select_penalty_by_delay_budget,
    step_response_delays,
)

__all__ = [
    "OnlineJumpDecoder",
    "compute_loss_matrix",
    "default_penalty_matrix",
    "fit_jump_model",
    "ConfirmationFilter",
    "PenaltyPoint",
    "ComparisonRow",
    "characteristic_loss_scale",
    "step_response_delays",
    "evaluate_penalty_curve",
    "select_penalty_by_delay_budget",
    "compare_carried_vs_rolling",
]
