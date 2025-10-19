"""Calculator for computing performance metrics from simulation results."""

import numpy as np
import polars as pl
from scipy import stats

from exploration_sequential_testing.metrics.models import (
    ErrorRateMetrics,
    PerformanceMetrics,
    StoppingTimeMetrics,
)
from exploration_sequential_testing.simulation.results import SimulationResults


class MetricsCalculator:
    """Calculator for performance metrics of sequential tests.

    This class computes various metrics from simulation results:
    - Type I and Type II error rates
    - Stopping time distributions
    - Average sample numbers
    - Efficiency compared to fixed designs
    """

    @staticmethod
    def compute_stopping_time_metrics(
        results: SimulationResults,
        quantiles: list[float] | None = None,
    ) -> StoppingTimeMetrics:
        """Compute stopping time metrics.

        Args:
            results: Simulation results
            quantiles: Quantiles to compute (default: [0.25, 0.5, 0.75, 0.9, 0.95])

        Returns:
            StoppingTimeMetrics
        """
        if quantiles is None:
            quantiles = [0.25, 0.5, 0.75, 0.9, 0.95]

        df = results.raw_results
        stopping_times = df["stopping_time"].to_numpy()

        # Compute statistics
        mean_st = float(np.mean(stopping_times))
        median_st = float(np.median(stopping_times))
        std_st = float(np.std(stopping_times))
        min_st = int(np.min(stopping_times))
        max_st = int(np.max(stopping_times))

        # Compute quantiles
        quantile_dict = {q: float(np.quantile(stopping_times, q)) for q in quantiles}

        return StoppingTimeMetrics(
            mean=mean_st,
            median=median_st,
            std=std_st,
            min=min_st,
            max=max_st,
            quantiles=quantile_dict,
        )

    @staticmethod
    def compute_error_metrics(
        null_results: SimulationResults | None = None,
        alt_results: SimulationResults | None = None,
    ) -> ErrorRateMetrics:
        """Compute error rate metrics.

        Args:
            null_results: Results under null hypothesis (for Type I error)
            alt_results: Results under alternative hypothesis (for Type II error)

        Returns:
            ErrorRateMetrics

        Raises:
            ValueError: If both results are None
        """
        if null_results is None and alt_results is None:
            raise ValueError(
                "At least one of null_results or alt_results must be provided"
            )

        # Type I error: P(reject H0 | H0 true)
        if null_results is not None:
            null_df = null_results.raw_results
            n_null = len(null_df)
            n_reject_under_null = len(null_df.filter(pl.col("decision") == "reject_h0"))
            n_continue_under_null = len(
                null_df.filter(pl.col("decision") == "continue")
            )

            type_i_error = n_reject_under_null / n_null if n_null > 0 else 0.0
            prob_reject_h0_under_null = type_i_error
            prob_continue_under_null = (
                n_continue_under_null / n_null if n_null > 0 else 0.0
            )
        else:
            type_i_error = 0.0
            prob_reject_h0_under_null = 0.0
            prob_continue_under_null = 0.0

        # Type II error: P(accept H0 | H1 true)
        # Power: P(reject H0 | H1 true)
        if alt_results is not None:
            alt_df = alt_results.raw_results
            n_alt = len(alt_df)
            n_accept_under_alt = len(alt_df.filter(pl.col("decision") == "accept_h0"))
            n_reject_under_alt = len(alt_df.filter(pl.col("decision") == "reject_h0"))
            n_continue_under_alt = len(alt_df.filter(pl.col("decision") == "continue"))

            type_ii_error = n_accept_under_alt / n_alt if n_alt > 0 else 0.0
            power = n_reject_under_alt / n_alt if n_alt > 0 else 0.0
            prob_accept_h0_under_alt = type_ii_error
            prob_continue_under_alt = n_continue_under_alt / n_alt if n_alt > 0 else 0.0
        else:
            type_ii_error = None
            power = None
            prob_accept_h0_under_alt = None
            prob_continue_under_alt = None

        return ErrorRateMetrics(
            type_i_error=type_i_error,
            type_ii_error=type_ii_error,
            power=power,
            prob_reject_h0_under_null=prob_reject_h0_under_null,
            prob_accept_h0_under_alt=prob_accept_h0_under_alt,
            prob_continue_under_null=prob_continue_under_null,
            prob_continue_under_alt=prob_continue_under_alt,
        )

    @staticmethod
    def compute_fixed_sample_size(
        alpha: float,
        beta: float,
        effect_size: float,
        test_type: str = "two_sided",
    ) -> int:
        """Compute fixed sample size for comparison.

        Uses standard formulas for Z-tests.

        Args:
            alpha: Type I error rate
            beta: Type II error rate
            effect_size: Standardized effect size (Cohen's d)
            test_type: "two_sided" or "one_sided"

        Returns:
            Required fixed sample size
        """
        z_alpha = (
            stats.norm.ppf(1 - alpha / 2)
            if test_type == "two_sided"
            else stats.norm.ppf(1 - alpha)
        )
        z_beta = stats.norm.ppf(1 - beta)

        # Formula: n = ((z_alpha + z_beta) / effect_size)^2
        n = ((z_alpha + z_beta) / effect_size) ** 2

        return int(np.ceil(n))

    @staticmethod
    def compute_efficiency(
        asn: float,
        fixed_n: int,
    ) -> float:
        """Compute sample size efficiency.

        Efficiency = ASN / N_fixed
        Values < 1.0 indicate the sequential test is more efficient.

        Args:
            asn: Average sample number (sequential test)
            fixed_n: Fixed sample size

        Returns:
            Efficiency ratio
        """
        return asn / fixed_n if fixed_n > 0 else float("inf")

    @staticmethod
    def compute_prob_exceed_fixed_n(
        results: SimulationResults,
        fixed_n: int,
    ) -> float:
        """Compute probability that stopping time exceeds fixed sample size.

        Args:
            results: Simulation results
            fixed_n: Fixed sample size

        Returns:
            Probability
        """
        df = results.raw_results
        stopping_times = df["stopping_time"].to_numpy()
        return float(np.mean(stopping_times > fixed_n))

    @classmethod
    def compute_full_metrics(
        cls,
        null_results: SimulationResults | None = None,
        alt_results: SimulationResults | None = None,
        alpha: float = 0.05,
        beta: float = 0.20,
        effect_size: float | None = None,
    ) -> PerformanceMetrics:
        """Compute all performance metrics.

        Args:
            null_results: Results under null hypothesis
            alt_results: Results under alternative hypothesis
            alpha: Type I error rate (for fixed sample size calculation)
            beta: Type II error rate (for fixed sample size calculation)
            effect_size: Standardized effect size (for fixed sample size calculation)

        Returns:
            PerformanceMetrics with all computed metrics
        """
        # Error metrics
        error_metrics = cls.compute_error_metrics(null_results, alt_results)

        # Stopping time metrics
        stopping_time_metrics_null = None
        stopping_time_metrics_alt = None
        asn_null = None
        asn_alt = None

        if null_results is not None:
            stopping_time_metrics_null = cls.compute_stopping_time_metrics(null_results)
            asn_null = stopping_time_metrics_null.mean

        if alt_results is not None:
            stopping_time_metrics_alt = cls.compute_stopping_time_metrics(alt_results)
            asn_alt = stopping_time_metrics_alt.mean

        # Update ASN in stopping time metrics
        if stopping_time_metrics_null is not None:
            stopping_time_metrics_null = StoppingTimeMetrics(
                **{**stopping_time_metrics_null.model_dump(), "asn_null": asn_null}
            )

        if stopping_time_metrics_alt is not None:
            stopping_time_metrics_alt = StoppingTimeMetrics(
                **{**stopping_time_metrics_alt.model_dump(), "asn_alt": asn_alt}
            )

        # Efficiency metrics
        fixed_n = None
        efficiency = None
        prob_exceed_fixed_n = None

        if effect_size is not None:
            fixed_n = cls.compute_fixed_sample_size(alpha, beta, effect_size)

            if asn_null is not None:
                efficiency = cls.compute_efficiency(asn_null, fixed_n)

            if null_results is not None:
                prob_exceed_fixed_n = cls.compute_prob_exceed_fixed_n(
                    null_results, fixed_n
                )

        # Get test name
        test_name = "unknown"
        if null_results is not None:
            test_name = null_results.test_name
        elif alt_results is not None:
            test_name = alt_results.test_name

        return PerformanceMetrics(
            test_name=test_name,
            error_metrics=error_metrics,
            stopping_time_metrics_null=stopping_time_metrics_null,
            stopping_time_metrics_alt=stopping_time_metrics_alt,
            efficiency=efficiency,
            prob_exceed_fixed_n=prob_exceed_fixed_n,
            fixed_sample_size=fixed_n,
        )
