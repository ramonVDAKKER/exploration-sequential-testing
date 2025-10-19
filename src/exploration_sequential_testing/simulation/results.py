"""Data structures for simulation results."""

from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from exploration_sequential_testing.tests.base import TestResult


class SimulationResults(BaseModel):
    """Results from a Monte Carlo simulation.

    Attributes:
        raw_results: Polars DataFrame with individual trajectory results
        test_name: Name of the test
        hypothesis: Which hypothesis data was generated under
        n_simulations: Number of completed simulations
        config_dict: Dictionary with simulation configuration
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_results: pl.DataFrame
    test_name: str
    hypothesis: str
    n_simulations: int = Field(ge=0)
    config_dict: dict[str, Any]

    @classmethod
    def from_test_results(
        cls,
        test_results: list[TestResult],
        hypothesis: str,
        config_dict: dict[str, Any],
    ) -> "SimulationResults":
        """Create SimulationResults from list of TestResult objects.

        Args:
            test_results: List of TestResult objects
            hypothesis: Which hypothesis was tested ("null" or "alternative")
            config_dict: Configuration dictionary

        Returns:
            SimulationResults object with Polars DataFrame
        """
        if not test_results:
            raise ValueError("test_results cannot be empty")

        # Convert to dictionary format
        data = []
        for i, result in enumerate(test_results):
            data.append(
                {
                    "simulation_id": i,
                    "hypothesis": hypothesis,
                    "decision": result.decision.value,
                    "stopping_time": result.stopping_time,
                    "test_name": result.test_name,
                    "trajectory": result.test_statistic_trajectory,
                    "observations": result.observations,
                }
            )

        # Create Polars DataFrame
        df = pl.DataFrame(data)

        return cls(
            raw_results=df,
            test_name=test_results[0].test_name,
            hypothesis=hypothesis,
            n_simulations=len(test_results),
            config_dict=config_dict,
        )

    def summary(self) -> dict[str, Any]:
        """Generate summary statistics.

        Returns:
            Dictionary with summary metrics
        """
        df = self.raw_results

        # Count decisions
        decision_counts = df.group_by("decision").agg(pl.count().alias("count"))

        # Stopping time statistics
        stopping_stats = df.select(
            [
                pl.col("stopping_time").mean().alias("mean_stopping_time"),
                pl.col("stopping_time").median().alias("median_stopping_time"),
                pl.col("stopping_time").std().alias("std_stopping_time"),
                pl.col("stopping_time").min().alias("min_stopping_time"),
                pl.col("stopping_time").max().alias("max_stopping_time"),
            ]
        ).to_dicts()[0]

        # Decision probabilities
        total = len(df)
        prob_reject = (
            len(df.filter(pl.col("decision") == "reject_h0")) / total
            if total > 0
            else 0.0
        )
        prob_accept = (
            len(df.filter(pl.col("decision") == "accept_h0")) / total
            if total > 0
            else 0.0
        )
        prob_continue = (
            len(df.filter(pl.col("decision") == "continue")) / total
            if total > 0
            else 0.0
        )

        return {
            "test_name": self.test_name,
            "hypothesis": self.hypothesis,
            "n_simulations": self.n_simulations,
            "decision_counts": decision_counts.to_dicts(),
            "prob_reject_h0": prob_reject,
            "prob_accept_h0": prob_accept,
            "prob_continue": prob_continue,
            **stopping_stats,
        }

    def to_csv(self, filepath: str) -> None:
        """Save results to CSV file.

        Args:
            filepath: Path to save CSV
        """
        # Convert list columns to strings for CSV
        df = self.raw_results.with_columns(
            [
                pl.col("trajectory").list.join(",").alias("trajectory"),
                pl.col("observations").list.join(",").alias("observations"),
            ]
        )
        df.write_csv(filepath)

    def to_parquet(self, filepath: str) -> None:
        """Save results to Parquet file (preserves list columns).

        Args:
            filepath: Path to save Parquet file
        """
        self.raw_results.write_parquet(filepath)

    @classmethod
    def from_parquet(cls, filepath: str) -> "SimulationResults":
        """Load results from Parquet file.

        Args:
            filepath: Path to Parquet file

        Returns:
            SimulationResults object
        """
        df = pl.read_parquet(filepath)

        # Extract metadata from DataFrame
        test_name = df["test_name"][0] if len(df) > 0 else "unknown"
        hypothesis = df["hypothesis"][0] if len(df) > 0 else "unknown"
        n_simulations = len(df)

        return cls(
            raw_results=df,
            test_name=test_name,
            hypothesis=hypothesis,
            n_simulations=n_simulations,
            config_dict={},
        )
