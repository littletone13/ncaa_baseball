from __future__ import annotations

from dataclasses import dataclass


# Shared run-scoring calibration applied to posterior intercepts.
SCORING_CALIBRATION = 0.12

# Policies for fatigue coverage contract handling.
FATIGUE_POLICY_ABORT = "abort"
FATIGUE_POLICY_DERISK = "de-risk"
FATIGUE_POLICY_IGNORE = "ignore"
FATIGUE_POLICY_CHOICES = (
    FATIGUE_POLICY_ABORT,
    FATIGUE_POLICY_DERISK,
    FATIGUE_POLICY_IGNORE,
)


def assert_scoring_calibration_parity(script_name: str, script_value: float) -> None:
    """Fail fast if a script-level calibration alias diverges from shared config."""
    if abs(float(script_value) - SCORING_CALIBRATION) > 1e-12:
        raise RuntimeError(
            f"{script_name}: SCORING_CALIBRATION divergence "
            f"(script={script_value}, shared={SCORING_CALIBRATION})"
        )


@dataclass(frozen=True)
class FatigueCoverageDecision:
    coverage: float
    required_teams: int
    covered_teams: int
    policy: str
    action: str
    message: str


def enforce_fatigue_coverage_policy(
    *,
    required_team_ids: set[str],
    fatigue_team_ids: set[str],
    policy: str,
    min_coverage: float,
    context_label: str,
) -> FatigueCoverageDecision:
    """
    Apply fatigue coverage contract for simulation/backtest workflows.

    Returns a decision describing whether fatigue should be applied as-is,
    ignored by policy, disabled (de-risk), or aborted.
    """
    if policy not in FATIGUE_POLICY_CHOICES:
        raise ValueError(f"Unknown fatigue policy: {policy}")

    threshold = float(min_coverage)
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"fatigue min coverage must be in [0,1], got {threshold}")

    required = {t.strip() for t in required_team_ids if t and str(t).strip()}
    covered = {t for t in required if t in fatigue_team_ids}
    missing = sorted(required - covered)
    req_n = len(required)
    cov_n = len(covered)
    coverage = 1.0 if req_n == 0 else cov_n / req_n

    base = (
        f"{context_label}: fatigue coverage {coverage:.1%} "
        f"({cov_n}/{req_n} teams, threshold={threshold:.1%}, policy={policy})"
    )
    if missing:
        sample = ", ".join(missing[:5])
        if len(missing) > 5:
            sample += ", ..."
        base = f"{base}; missing={sample}"

    if policy == FATIGUE_POLICY_IGNORE:
        return FatigueCoverageDecision(
            coverage=coverage,
            required_teams=req_n,
            covered_teams=cov_n,
            policy=policy,
            action="ignore",
            message=f"{base}; applying available fatigue rows (partial coverage allowed).",
        )

    if coverage >= threshold:
        return FatigueCoverageDecision(
            coverage=coverage,
            required_teams=req_n,
            covered_teams=cov_n,
            policy=policy,
            action="apply",
            message=f"{base}; coverage OK, applying fatigue adjustments.",
        )

    if policy == FATIGUE_POLICY_ABORT:
        raise RuntimeError(f"{base}; aborting due to low fatigue coverage.")

    # de-risk
    return FatigueCoverageDecision(
        coverage=coverage,
        required_teams=req_n,
        covered_teams=cov_n,
        policy=policy,
        action="de-risk",
        message=f"{base}; de-risking by disabling fatigue adjustments.",
    )
