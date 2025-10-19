"""Factory for creating distributions from configuration."""

from typing import Any

from exploration_sequential_testing.distributions.base import Distribution
from exploration_sequential_testing.distributions.bernoulli import (
    BernoulliDistribution,
    BernoulliParams,
)
from exploration_sequential_testing.distributions.gaussian import (
    GaussianDistribution,
    GaussianParams,
)


class DistributionFactory:
    """Factory for creating distribution instances from configuration.

    Example:
        >>> config = {"type": "gaussian", "mean": 0.5, "std": 1.0}
        >>> dist = DistributionFactory.create(config)
        >>> samples = dist.sample(100, rng)
    """

    _registry: dict[str, type[Distribution]] = {
        "bernoulli": BernoulliDistribution,
        "gaussian": GaussianDistribution,
        "normal": GaussianDistribution,  # Alias
    }

    _params_registry: dict[str, type] = {
        "bernoulli": BernoulliParams,
        "gaussian": GaussianParams,
        "normal": GaussianParams,  # Alias
    }

    @classmethod
    def create(cls, config: dict[str, Any]) -> Distribution:
        """Create a distribution from configuration dictionary.

        Args:
            config: Dictionary with keys:
                - "type": Distribution type (e.g., "gaussian", "bernoulli")
                - Other keys are distribution-specific parameters

        Returns:
            Distribution instance

        Raises:
            ValueError: If distribution type is not recognized
            ValidationError: If parameters are invalid (from Pydantic)

        Example:
            >>> config = {"type": "bernoulli", "p": 0.3}
            >>> dist = DistributionFactory.create(config)
        """
        dist_type = config.get("type", "").lower()

        if dist_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown distribution type: '{dist_type}'. "
                f"Available types: {available}"
            )

        # Get the distribution and params classes
        dist_class = cls._registry[dist_type]
        params_class = cls._params_registry[dist_type]

        # Extract parameters (all keys except 'type')
        params_dict = {k: v for k, v in config.items() if k != "type"}

        # Create params (Pydantic will validate)
        params = params_class(**params_dict)

        # Create and return distribution
        return dist_class(params)

    @classmethod
    def register(
        cls,
        name: str,
        dist_class: type[Distribution],
        params_class: type,
    ) -> None:
        """Register a new distribution type.

        This allows users to add custom distributions.

        Args:
            name: Name to use in configuration
            dist_class: Distribution class
            params_class: Parameters class (Pydantic model)

        Example:
            >>> DistributionFactory.register("poisson", PoissonDistribution, PoissonParams)
        """
        cls._registry[name.lower()] = dist_class
        cls._params_registry[name.lower()] = params_class

    @classmethod
    def available_distributions(cls) -> list[str]:
        """Get list of available distribution types.

        Returns:
            List of registered distribution names
        """
        return sorted(cls._registry.keys())
