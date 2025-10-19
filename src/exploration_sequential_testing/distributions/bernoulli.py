"""Bernoulli distribution for sequential testing."""

import numpy as np
from pydantic import Field, field_validator

from exploration_sequential_testing.distributions.base import (
    Distribution,
    DistributionParams,
)


class BernoulliParams(DistributionParams):
    """Parameters for Bernoulli distribution.

    Attributes:
        p: Success probability, must be in (0, 1)
    """

    p: float = Field(gt=0.0, lt=1.0, description="Success probability")

    @field_validator("p")
    @classmethod
    def validate_probability(cls, v: float) -> float:
        """Ensure probability is strictly between 0 and 1."""
        if not 0.0 < v < 1.0:
            raise ValueError(f"Probability must be in (0, 1), got {v}")
        return v


class BernoulliDistribution(Distribution):
    """Bernoulli distribution for binary outcome sequential testing."""

    def __init__(self, params: BernoulliParams):
        """Initialize Bernoulli distribution.

        Args:
            params: Bernoulli parameters (success probability p)
        """
        super().__init__(params)
        self.params: BernoulliParams

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Generate n Bernoulli samples.

        Args:
            n: Number of samples
            rng: Random number generator

        Returns:
            Array of 0s and 1s of shape (n,)
        """
        return rng.binomial(n=1, p=self.params.p, size=n)

    def log_likelihood(self, data: np.ndarray) -> float:
        """Compute log-likelihood of data under current parameters.

        For Bernoulli: log L = Σ_i [x_i * log(p) + (1 - x_i) * log(1 - p)]

        Args:
            data: Binary data (0s and 1s)

        Returns:
            Log-likelihood value
        """
        p = self.params.p
        # Handle edge cases to avoid log(0)
        log_p = np.log(p)
        log_1_p = np.log(1 - p)

        return float(np.sum(data * log_p + (1 - data) * log_1_p))

    def log_likelihood_ratio(
        self, data: np.ndarray, alt_params: BernoulliParams
    ) -> float:
        """Compute log-likelihood ratio for SPRT.

        LLR = log(L(data | p1) / L(data | p0))
            = Σ_i [x_i * log(p1/p0) + (1 - x_i) * log((1-p1)/(1-p0))]

        Args:
            data: Binary data
            alt_params: Alternative hypothesis parameters

        Returns:
            Log-likelihood ratio
        """
        p0 = self.params.p  # Null hypothesis
        p1 = alt_params.p  # Alternative hypothesis

        # Log-likelihood ratio per observation
        log_ratio_1 = np.log(p1 / p0)
        log_ratio_0 = np.log((1 - p1) / (1 - p0))

        return float(np.sum(data * log_ratio_1 + (1 - data) * log_ratio_0))

    @property
    def name(self) -> str:
        """Return distribution name."""
        return "Bernoulli"

    def to_dict(self) -> dict[str, any]:
        """Serialize distribution to dictionary.

        Returns:
            Dictionary compatible with DistributionFactory.create()
        """
        result = {"type": "bernoulli"}
        result.update(self.params.model_dump())
        return result

    def __repr__(self) -> str:
        """Return string representation."""
        return f"BernoulliDistribution(p={self.params.p:.4f})"
