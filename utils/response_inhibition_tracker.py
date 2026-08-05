from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Optional
from collections import deque


@dataclass
class TrialUpdate:
    """Container for one trial-level adaptive update."""

    trial_type: str
    current_ssd_ms: float
    move_summary: Optional[float] = None
    emg_summary: Optional[float] = None
    go_rt_ms: Optional[float] = None
    stop_success: Optional[bool] = None
    next_ssd_ms: Optional[float] = None
    decision: str = "keep"


class ResponseInhibitionTracker:
    """Track and update the adaptive stop-signal delay (SSD).

    The tracker keeps the current SSD and applies a 1-up / 1-down update
    rule after each STOP/abort trial. It also stores recent SSD values so
    the experiment loop can check convergence and stop criteria online.
    """

    def __init__(
        self,
        initial_ssd_ms: float,
        step_size_ms: float,
        min_ssd_ms: float,
        max_ssd_ms: float,
        stop_window_size: int = 15,
        convergence_sd_ms: float = 30.0,
        go_rt_limit_ms: float = 800.0,
    ):
        self.initial_ssd_ms = float(initial_ssd_ms)
        self.current_ssd_ms = float(initial_ssd_ms)
        self.step_size_ms = float(step_size_ms)
        self.min_ssd_ms = float(min_ssd_ms)
        self.max_ssd_ms = float(max_ssd_ms)
        self.stop_window_size = int(stop_window_size)
        self.convergence_sd_ms = float(convergence_sd_ms)
        self.go_rt_limit_ms = float(go_rt_limit_ms)
        self._recent_stop_ssd_ms: Deque[float] = deque(maxlen=self.stop_window_size)

    def get_current_ssd_ms(self) -> float:
        return self.current_ssd_ms

    def record_stop_trial(self, stop_success: bool) -> TrialUpdate:
        """Apply the 1-up / 1-down rule to the current SSD.

        STOP_SUCCESS -> increase SSD
        STOP_FAILURE -> decrease SSD
        """
        previous_ssd = self.current_ssd_ms

        if stop_success:
            updated_ssd = previous_ssd + self.step_size_ms
            decision = "lengthen"
        else:
            updated_ssd = previous_ssd - self.step_size_ms
            decision = "shorten"

        updated_ssd = self._clip_ssd(updated_ssd)
        self.current_ssd_ms = updated_ssd
        self._recent_stop_ssd_ms.append(updated_ssd)

        return TrialUpdate(
            trial_type="STOP",
            current_ssd_ms=previous_ssd,
            stop_success=stop_success,
            next_ssd_ms=updated_ssd,
            decision=decision,
        )

    def record_stop_trial_from_summary(
        self,
        move_summary: Optional[float],
        move_threshold: Optional[float],
        emg_summary: Optional[float] = None,
    ) -> TrialUpdate:
        """Apply the update rule from an ACC movement summary.

        STOP_SUCCESS is defined as move_summary < move_threshold.
        When the summary or threshold is missing, the tracker keeps the
        current SSD unchanged and returns a 'keep' decision.
        """
        previous_ssd = self.current_ssd_ms

        if move_summary is None or move_threshold is None or move_summary != move_summary:
            return TrialUpdate(
                trial_type="STOP",
                current_ssd_ms=previous_ssd,
                move_summary=move_summary,
                emg_summary=emg_summary,
                stop_success=None,
                next_ssd_ms=previous_ssd,
                decision="keep",
            )

        stop_success = move_summary < move_threshold
        stop_update = self.record_stop_trial(stop_success=stop_success)
        stop_update.move_summary = move_summary
        stop_update.emg_summary = emg_summary
        return stop_update

    def record_go_trial(self, go_rt_ms: float) -> TrialUpdate:
        """Apply the PD guardrail for slow GO responses.

        GO trials do not update SSD. If the RT is above the configured
        guardrail, the trial is flagged as too slow and ignored for SSD
        adaptation.
        """
        too_slow = go_rt_ms > self.go_rt_limit_ms
        decision = "too_slow" if too_slow else "keep"

        return TrialUpdate(
            trial_type="GO",
            current_ssd_ms=self.current_ssd_ms,
            go_rt_ms=go_rt_ms,
            stop_success=None,
            next_ssd_ms=self.current_ssd_ms,
            decision=decision,
        )

    def has_converged(self) -> bool:
        """Check whether the recent STOP SSD values have stabilized."""
        if len(self._recent_stop_ssd_ms) < self.stop_window_size:
            return False

        mean_ssd = sum(self._recent_stop_ssd_ms) / len(self._recent_stop_ssd_ms)
        variance = sum((value - mean_ssd) ** 2 for value in self._recent_stop_ssd_ms) / len(
            self._recent_stop_ssd_ms
        )
        return variance ** 0.5 < self.convergence_sd_ms

    def _clip_ssd(self, ssd_ms: float) -> float:
        return max(self.min_ssd_ms, min(self.max_ssd_ms, ssd_ms))