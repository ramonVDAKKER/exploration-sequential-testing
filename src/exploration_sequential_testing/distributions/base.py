"""Base classes for distributions."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict


class DistributionParams(BaseModel):
    """Base class for distribution parameters.

    Uses Pydantic for validation and serialization.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class Distribution(ABC):
    """Abstract base class for data generating distributions.

    All distributions must implement methods for sampling and computing
    log-likelihood, which are essential for multiple sequential tests.
    """

    def __init__(self, params: DistributionParams):
        """Initialize distribution with parameters.

        Args:
            params: Distribution parameters (validated by Pydantic)
        """
        self.params = params

    @abstractmethod
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Generate n independent samples from the distribution.

        Args:
            n: Number of samples to generate
            rng: NumPy random generator for reproducibility

        Returns:
            Array of shape (n,) with samples
        """
        pass

    @abstractmethod
    def log_likelihood(self, data: np.ndarray) -> float:
        """Compute log-likelihood of data under current parameters.

        Args:
            data: Observed data array

        Returns:
            Log-likelihood value
        """
        pass

    @abstractmethod
    def log_likelihood_ratio(
        self,
        data: np.ndarray,
        alt_params: DistributionParams
    ) -> float:
        """Compute log-likelihood ratio between alternative and current (null) params.

        This is the core computation for SPRT and related tests:
        LLR = log(L(data | alt_params) / L(data | null_params))

        Args:
            data: Observed data array
            alt_params: Alternative hypothesis parameters

        Returns:
            Log-likelihood ratio
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return human-readable name of distribution."""
        pass

    def to_dict(self) -> dict[str, Any]:
        """Serialize distribution to dictionary.

        Returns:
            Dictionary compatible with DistributionFactory.create().
            Format: {"type": "...", "param1": ..., "param2": ...}
            
        Note:
            Subclasses should override this to provide the correct type key.
        """
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement to_dict()"
        )
