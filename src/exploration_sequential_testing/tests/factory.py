"""Factory for creating sequential tests from configuration."""

from typing import Any

from exploration_sequential_testing.distributions.base import Distribution
from exploration_sequential_testing.tests.base import SequentialTest, TestConfig
from exploration_sequential_testing.tests.group_sequential import (
    GroupSequentialConfig,
    GroupSequentialTest,
)
from exploration_sequential_testing.tests.sprt import SPRT, SPRTConfig


class TestFactory:
    """Factory for creating sequential test instances from configuration.

    Example:
        >>> config = {
        ...     "type": "sprt",
        ...     "alpha": 0.05,
        ...     "beta": 0.20,
        ...     "truncate_at": 1_000
        ... }
        >>> test = TestFactory.create(config, null_dist, alt_dist)
    """

    _registry: dict[str, type[Any]] = {
        "sprt": SPRT,
        "group_sequential": GroupSequentialTest,
    }

    _config_registry: dict[str, type[TestConfig]] = {
        "sprt": SPRTConfig,
        "group_sequential": GroupSequentialConfig,
    }

    @classmethod
    def create(
        cls,
        config: dict[str, Any],
        null_dist: Distribution,
        alt_dist: Distribution,
    ) -> SequentialTest:
        """Create a sequential test from configuration.

        Args:
            config: Dictionary with keys:
                - "type": Test type (e.g., "sprt", "group_sequential")
                - Other keys are test-specific parameters
            null_dist: Distribution under null hypothesis
            alt_dist: Distribution under alternative hypothesis

        Returns:
            SequentialTest instance

        Raises:
            ValueError: If test type is not recognized
            ValidationError: If parameters are invalid (from Pydantic)

        Example:
            >>> config = {"type": "sprt", "alpha": 0.05, "beta": 0.20}
            >>> test = TestFactory.create(config, null_dist, alt_dist)
        """
        test_type = config.get("type", "").lower()

        if test_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown test type: '{test_type}'. Available types: {available}"
            )

        # Get the test and config classes
        test_class = cls._registry[test_type]
        config_class = cls._config_registry[test_type]

        # Extract parameters (all keys except 'type')
        params_dict = {k: v for k, v in config.items() if k != "type"}

        # Create config (Pydantic will validate)
        test_config = config_class(**params_dict)

        # Create and return test
        return test_class(test_config, null_dist, alt_dist)

    @classmethod
    def register(
        cls,
        name: str,
        test_class: type[Any],
        config_class: type[TestConfig],
    ) -> None:
        """Register a new test type.

        This allows users to add custom tests.

        Args:
            name: Name to use in configuration
            test_class: Test class
            config_class: Configuration class (Pydantic model)

        Example:
            >>> TestFactory.register("custom_test", CustomTest, CustomConfig)
        """
        cls._registry[name.lower()] = test_class
        cls._config_registry[name.lower()] = config_class

    @classmethod
    def available_tests(cls) -> list[str]:
        """Get list of available test types.

        Returns:
            List of registered test names
        """
        return sorted(cls._registry.keys())
