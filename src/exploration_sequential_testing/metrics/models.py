"""Pydantic models for metrics."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StoppingTimeMetrics(BaseModel):
    """Metrics related to stopping times.

    Attributes:
        mean: Mean stopping time
        median: Median stopping time
        std: Standard deviation of stopping time
        min: Minimum stopping time
        max: Maximum stopping time
        quantiles: Dictionary of quantiles (e.g., {0.25: 50, 0.75: 150})
        asn_null: Average sample number under H0
        asn_alt: Average sample number under H1
    """

    model_config = ConfigDict(frozen=True)

    mean: float = Field(ge=0.0)
    median: float = Field(ge=0.0)
    std: float = Field(ge=0.0)
    min: int = Field(ge=0)
    max: int = Field(ge=0)
    quantiles: dict[float, float] = Field(default_factory=dict)
    asn_null: float | None = None
    asn_alt: float | None = None


class ErrorRateMetrics(BaseModel):
    """Metrics related to Type I and Type II errors.

    Attributes:
        type_I_error: Empirical Type I error rate (should be ≤ α)
        type_II_error: Empirical Type II error rate (should be ≤ β)
        power: Empirical power (1 - Type II error)
        prob_reject_h0_under_null: P(reject H0 | H0 true) = Type I error
        prob_accept_h0_under_alt: P(accept H0 | H1 true) = Type II error
        prob_continue_under_null: P(no decision | H0 true)
        prob_continue_under_alt: P(no decision | H1 true)
    """

    model_config = ConfigDict(frozen=True)

    type_i_error: float = Field(ge=0.0, le=1.0)
    type_ii_error: float | None = Field(default=None, ge=0.0, le=1.0)
    power: float | None = Field(default=None, ge=0.0, le=1.0)
    prob_reject_h0_under_null: float = Field(ge=0.0, le=1.0)
    prob_accept_h0_under_alt: float | None = Field(default=None, ge=0.0, le=1.0)
    prob_continue_under_null: float = Field(ge=0.0, le=1.0)
    prob_continue_under_alt: float | None = Field(default=None, ge=0.0, le=1.0)


class PerformanceMetrics(BaseModel):
    """Complete performance metrics for a sequential test.

    Attributes:
        test_name: Name of the test
        error_metrics: Error rate metrics
        stopping_time_metrics_null: Stopping time metrics under H0
        stopping_time_metrics_alt: Stopping time metrics under H1 (if available)
        efficiency: Sample size efficiency (ASN / fixed_sample_size)
        prob_exceed_fixed_n: Probability of exceeding fixed sample size
        fixed_sample_size: Reference fixed sample size for comparison
    """

    model_config = ConfigDict(frozen=True)

    test_name: str
    error_metrics: ErrorRateMetrics
    stopping_time_metrics_null: StoppingTimeMetrics
    stopping_time_metrics_alt: StoppingTimeMetrics | None = None
    efficiency: float | None = Field(default=None, ge=0.0)
    prob_exceed_fixed_n: float | None = Field(default=None, ge=0.0, le=1.0)
    fixed_sample_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary for DataFrame."""
        result = {
            "test_name": self.test_name,
            # Error metrics
            "type_i_error": self.error_metrics.type_i_error,
            "type_ii_error": self.error_metrics.type_ii_error,
            "power": self.error_metrics.power,
            "prob_reject_h0_under_null": self.error_metrics.prob_reject_h0_under_null,
            "prob_continue_under_null": self.error_metrics.prob_continue_under_null,
            # Stopping time under null
            "asn_null": self.stopping_time_metrics_null.mean,
            "median_stopping_time_null": self.stopping_time_metrics_null.median,
            "std_stopping_time_null": self.stopping_time_metrics_null.std,
            # Efficiency metrics
            "efficiency": self.efficiency,
            "prob_exceed_fixed_n": self.prob_exceed_fixed_n,
            "fixed_sample_size": self.fixed_sample_size,
        }

        # Add alternative metrics if available
        if self.stopping_time_metrics_alt is not None:
            result["asn_alt"] = self.stopping_time_metrics_alt.mean
            result["median_stopping_time_alt"] = self.stopping_time_metrics_alt.median
            result["std_stopping_time_alt"] = self.stopping_time_metrics_alt.std

        if self.error_metrics.prob_accept_h0_under_alt is not None:
            result["prob_accept_h0_under_alt"] = (
                self.error_metrics.prob_accept_h0_under_alt
            )

        if self.error_metrics.prob_continue_under_alt is not None:
            result["prob_continue_under_alt"] = (
                self.error_metrics.prob_continue_under_alt
            )

        return result

    def summary_string(self) -> str:
        """Generate a human-readable summary string."""
        lines = [
            f"=== {self.test_name} Performance Metrics ===",
            "",
            "Error Rates:",
            f"  Type I Error:  {self.error_metrics.type_i_error:.4f}",
        ]

        if self.error_metrics.type_ii_error is not None:
            lines.append(f"  Type II Error: {self.error_metrics.type_ii_error:.4f}")

        if self.error_metrics.power is not None:
            lines.append(f"  Power:         {self.error_metrics.power:.4f}")

        lines.extend(
            [
                "",
                "Stopping Times (under H0):",
                f"  Mean (ASN):    {self.stopping_time_metrics_null.mean:.2f}",
                f"  Median:        {self.stopping_time_metrics_null.median:.2f}",
                f"  Std Dev:       {self.stopping_time_metrics_null.std:.2f}",
            ]
        )

        if self.stopping_time_metrics_alt is not None:
            lines.extend(
                [
                    "",
                    "Stopping Times (under H1):",
                    f"  Mean (ASN):    {self.stopping_time_metrics_alt.mean:.2f}",
                    f"  Median:        {self.stopping_time_metrics_alt.median:.2f}",
                    f"  Std Dev:       {self.stopping_time_metrics_alt.std:.2f}",
                ]
            )

        if self.efficiency is not None:
            lines.extend(
                [
                    "",
                    f"Efficiency:      {self.efficiency:.4f}",
                ]
            )

        if self.prob_exceed_fixed_n is not None:
            lines.extend(
                [
                    f"P(N > N_fixed): {self.prob_exceed_fixed_n:.4f}",
                ]
            )

        return "\n".join(lines)
