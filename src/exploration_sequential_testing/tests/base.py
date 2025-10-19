"""Base classes for sequential tests."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class Decision(str, Enum):
    """Possible decisions in sequential testing."""

    CONTINUE = "continue"  # Continue sampling
    REJECT_H0 = "reject_null"
    ACCEPT_H0 = "accept_null"


class TestConfig(BaseModel):
    """Base configuration for sequential tests.

    Attributes:
        alpha: Type I error rate (significance level)
        beta: Type II error rate (1 - power)
    """

    model_config = ConfigDict(frozen=True)

    alpha: float = Field(gt=0.0, lt=1.0, default=0.05, description="Type I error rate")
    beta: float = Field(gt=0.0, lt=1.0, default=0.20, description="Type II error rate")


class TestResult(BaseModel):
    """Result from a single sequential test trajectory.

    Attributes:
        decision: Final decision (reject_h0, accept_h0, or continue if truncated)
        stopping_time: Number of observations until decision
        test_statistic_trajectory: Evolution of test statistic over time
        observations: The observed data
        test_name: Name of the test
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    decision: Decision
    stopping_time: int = Field(ge=1)
    test_statistic_trajectory: list[float]
    observations: list[float]
    test_name: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "decision": self.decision.value,
            "stopping_time": self.stopping_time,
            "test_statistic_trajectory": self.test_statistic_trajectory,
            "observations": self.observations,
            "test_name": self.test_name,
        }


class SequentialTest(ABC):
    """Abstract base class for sequential tests.

    A sequential test processes observations one at a time (or in batches)
    and makes a decision to:
    1. Continue sampling
    2. Reject the null hypothesis
    3. Accept the null hypothesis

    Subclasses must implement the specific test logic.
    """

    def __init__(self, config: TestConfig):
        """Initialize sequential test.

        Args:
            config: Test configuration (alpha, beta, etc.)
        """
        self.config = config
        self._n_observations = 0
        self._test_statistic_history: list[float] = []
        self._observations_history: list[float] = []

    @abstractmethod
    def reset(self) -> None:
        """Reset test state for a new trajectory.

        This should clear any accumulated statistics and set the test
        back to its initial state.
        """
        self._n_observations = 0
        self._test_statistic_history = []
        self._observations_history = []

    @abstractmethod
    def update(self, observation: float) -> Decision:
        """Update test with a new observation and return decision.

        Args:
            observation: New data point

        Returns:
            Decision (continue, reject_h0, or accept_h0)
        """
        pass

    @abstractmethod
    def get_boundaries(self, n_max: int) -> tuple[np.ndarray, np.ndarray]:
        """Get decision boundaries for plotting.

        Args:
            n_max: Maximum sample size to compute boundaries for

        Returns:
            Tuple of (upper_boundary, lower_boundary) arrays of shape (n_max,)
            Upper boundary: reject H0 if test statistic exceeds this
            Lower boundary: accept H0 if test statistic falls below this
        """
        pass

    @abstractmethod
    def get_test_statistic(self) -> float:
        """Get current value of the test statistic.

        Returns:
            Current test statistic value
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return human-readable name of the test."""
        pass

    def get_result(self, decision: Decision) -> TestResult:
        """Package current state into a TestResult.

        Args:
            decision: Final decision

        Returns:
            TestResult object
        """
        return TestResult(
            decision=decision,
            stopping_time=self._n_observations,
            test_statistic_trajectory=self._test_statistic_history.copy(),
            observations=self._observations_history.copy(),
            test_name=self.name,
        )

    def run_trajectory(
        self,
        data_generator: Callable,
        max_n: int = 1_000,
    ) -> TestResult:
        """Run a complete test trajectory.

        Args:
            data_generator: Callable that returns one observation when called
            max_n: Maximum number of observations before forced stop

        Returns:
            TestResult with final decision and trajectory
        """
        self.reset()

        for _ in range(max_n):
            obs = data_generator()
            decision = self.update(obs)

            if decision != Decision.CONTINUE:
                return self.get_result(decision)

        # Truncated - reached max_n without decision
        return self.get_result(Decision.CONTINUE)

    def to_dict(self) -> dict[str, Any]:
        """Serialize test configuration to dictionary.

        Returns:
            Dictionary with test type and configuration parameters.
            Format compatible with TestFactory.create().

        Note:
            Subclasses should override this to provide the correct type key.
        """
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement to_dict()"
        )
