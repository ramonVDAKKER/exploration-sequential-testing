"""Sequential Probability Ratio Test (SPRT).

References:
    https://en.wikipedia.org/wiki/Sequential_probability_ratio_test
"""

from typing import Any

import numpy as np
from pydantic import Field

from exploration_sequential_testing.distributions.base import Distribution
from exploration_sequential_testing.tests.base import (
    Decision,
    SequentialTest,
    TestConfig,
)


class SPRTConfig(TestConfig):
    """Configuration for SPRT.

    Attributes:
        alpha: Type I error rate
        beta: Type II error rate
        truncate_at: Optional maximum sample size (None = no truncation)
    """

    truncate_at: int | None = Field(
        default=None,
        ge=2,
        description="Maximum sample size (None for no truncation)",
    )


class SPRT(SequentialTest):
    """Sequential Probability Ratio Test.

    The SPRT computes the log-likelihood ratio (LLR) between two simple hypotheses:
    - H_0: data ~ Distribution(null_params)
    - H_1: data ~ Distribution(alt_params)

    Decision boundaries (https://en.wikipedia.org/wiki/Sequential_probability_ratio_test):
    - Reject H_0 if LLR >= log(B) = log((1-β)/α)
    - Accept H_0 if LLR <= log(A) = log(β/(1-α))
    - Continue if log(A) < LLR < log(B)

    The SPRT minimizes the expected sample size under both hypotheses
    among all tests with the same error rates.
    """

    def __init__(
        self,
        config: SPRTConfig,
        null_dist: Distribution,
        alt_dist: Distribution,
    ):
        """Initialize SPRT.

        Args:
            config: SPRT configuration
            null_dist: Distribution under null hypothesis
            alt_dist: Distribution under alternative hypothesis
        """
        super().__init__(config)
        self.config: SPRTConfig
        self.null_dist = null_dist
        self.alt_dist = alt_dist

        # Compute decision boundaries using Wald's approximation
        self.log_A = np.log(config.beta / (1 - config.alpha))
        self.log_B = np.log((1 - config.beta) / config.alpha)

        # Initialize test statistic (cumulative log-likelihood ratio)
        self._llr = 0.0

    def reset(self) -> None:
        """Reset test state for new trajectory."""
        super().reset()
        self._llr = 0.0

    def update(self, observation: float) -> Decision:
        """Update SPRT with new observation.

        Args:
            observation: New data point

        Returns:
            Decision based on current LLR
        """
        self._n_observations += 1
        self._observations_history.append(observation)

        # Compute log-likelihood ratio for this observation
        # LLR = log(L(obs | H1) / L(obs | H0))
        obs_array = np.array([observation])
        ll_alt = self.alt_dist.log_likelihood(obs_array)
        ll_null = self.null_dist.log_likelihood(obs_array)
        llr_increment = ll_alt - ll_null

        # Update cumulative LLR
        self._llr += llr_increment
        self._test_statistic_history.append(self._llr)

        # Check truncation
        if (
            self.config.truncate_at is not None
            and self._n_observations >= self.config.truncate_at
        ):
            # Force decision at truncation point
            if self._llr >= 0:
                return Decision.REJECT_H0
            else:
                return Decision.ACCEPT_H0

        # Make decision based on boundaries
        if self._llr >= self.log_B:
            return Decision.REJECT_H0
        elif self._llr <= self.log_A:
            return Decision.ACCEPT_H0
        else:
            return Decision.CONTINUE

    def get_test_statistic(self) -> float:
        """Get current log-likelihood ratio.

        Returns:
            Current cumulative LLR
        """
        return self._llr

    def get_boundaries(self, n_max: int) -> tuple[np.ndarray, np.ndarray]:
        """Get SPRT decision boundaries.

        For SPRT, boundaries are constant over time (horizontal lines).

        Args:
            n_max: Maximum sample size

        Returns:
            (upper_boundary, lower_boundary) where:
            - upper_boundary: array of log(B) values
            - lower_boundary: array of log(A) values
        """
        upper = np.full(n_max, self.log_B)
        lower = np.full(n_max, self.log_A)
        return upper, lower

    @property
    def name(self) -> str:
        """Return test name."""
        return "SPRT"

    def to_dict(self) -> dict[str, Any]:
        """Serialize test configuration to dictionary.

        Returns:
            Dictionary compatible with TestFactory.create()
        """
        result = {"type": "sprt"}
        result.update(self.config.model_dump())
        return result

    def expected_sample_size(self, true_params: Distribution) -> float:
        """Compute expected sample size under given distribution.

        This uses Wald's equation for the expected sample size.

        Args:
            true_params: True distribution generating the data

        Returns:
            Expected sample size (Average Sample Number)
        """
        # Generate a sample to estimate E[LLR increment]
        rng = np.random.default_rng(42)
        sample = true_params.sample(10_000, rng)

        # Compute expected LLR increment per observation
        ll_alt = self.alt_dist.log_likelihood(sample) / len(sample)
        ll_null = self.null_dist.log_likelihood(sample) / len(sample)
        expected_llr_increment = ll_alt - ll_null

        if abs(expected_llr_increment) < 1e-10:
            # At indifference point
            return float("inf")

        # Wald's equation: E[N] ≈ E[LLR at stopping] / E[LLR increment]
        # Probability of rejecting H0
        p_reject = (
            (1 - self.config.beta) if expected_llr_increment > 0 else self.config.alpha
        )
        expected_llr_at_stopping = p_reject * self.log_B + (1 - p_reject) * self.log_A

        asn = expected_llr_at_stopping / expected_llr_increment

        return float(asn)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"SPRT(α={self.config.alpha:.4f}, β={self.config.beta:.4f}, "
            f"log_A={self.log_A:.4f}, log_B={self.log_B:.4f})"
        )
