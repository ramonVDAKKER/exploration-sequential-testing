"""Gaussian (Normal) distribution for sequential testing."""

from typing import Any

import numpy as np
from pydantic import Field

from exploration_sequential_testing.distributions.base import (
    Distribution,
    DistributionParams,
)


class GaussianParams(DistributionParams):
    """Parameters for Gaussian distribution.

    Attributes:
        mean: Mean (location parameter)
        std: Standard deviation (scale parameter), must be positive
    """

    mean: float = Field(description="Mean of the distribution")
    std: float = Field(gt=0.0, description="Standard deviation (must be positive)")


class GaussianDistribution(Distribution):
    """Gaussian (Normal) distribution for continuous sequential testing."""

    def __init__(self, params: GaussianParams):
        """Initialize Gaussian distribution.

        Args:
            params: Gaussian parameters (mean, std)
        """
        super().__init__(params)
        self.params: GaussianParams

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Generate n Gaussian samples.

        Args:
            n: Number of samples
            rng: Random number generator

        Returns:
            Array of shape (n,) with Gaussian samples
        """
        return rng.normal(loc=self.params.mean, scale=self.params.std, size=n)

    def log_likelihood(self, data: np.ndarray) -> float:
        """Compute log-likelihood of data under current parameters.

        For Gaussian: log L = -n/2 * log(2π) - n * log(σ) - Σ(x_i - μ)^2 / (2σ^2)

        Args:
            data: Continuous data

        Returns:
            Log-likelihood value
        """
        n = len(data)
        mu = self.params.mean
        sigma = self.params.std

        # Compute log-likelihood
        log_likelihood = (
            -0.5 * n * np.log(2 * np.pi)
            - n * np.log(sigma)
            - np.sum((data - mu) ** 2) / (2 * sigma**2)
        )

        return float(log_likelihood)

    def log_likelihood_ratio(
        self, data: np.ndarray, alt_params: GaussianParams
    ) -> float:
        """Compute log-likelihood ratio for SPRT.

        For Gaussian with known variance:
        LLR = Σ[(μ1 - μ0) * x_i / σ^2 - (μ1^2 - μ0^2) / (2σ^2)]

        For Gaussian with different variances:
        LLR = log(L(data | μ1, σ1) / L(data | μ0, σ0))

        Args:
            data: Continuous data
            alt_params: Alternative hypothesis parameters

        Returns:
            Log-likelihood ratio
        """
        mu0, sigma0 = self.params.mean, self.params.std
        mu1, sigma1 = alt_params.mean, alt_params.std

        n = len(data)

        # Full log-likelihood ratio (handles different variances)
        llr = (
            -n * np.log(sigma1 / sigma0)
            - np.sum((data - mu1) ** 2) / (2 * sigma1**2)
            + np.sum((data - mu0) ** 2) / (2 * sigma0**2)
        )

        return float(llr)

    @property
    def name(self) -> str:
        """Return distribution name."""
        return "Gaussian"

    def to_dict(self) -> dict[str, Any]:
        """Serialize distribution to dictionary.

        Returns:
            Dictionary compatible with DistributionFactory.create()
        """
        result = {"type": "gaussian"}
        result.update(self.params.model_dump())
        return result

    def __repr__(self) -> str:
        """Return string representation."""
        return f"GaussianDistribution(mean={self.params.mean:.4f}, std={self.params.std:.4f})"
