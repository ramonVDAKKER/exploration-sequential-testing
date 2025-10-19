"""Group Sequential Tests with various boundary functions.

Group sequential tests perform interim analyses at pre-specified "looks"
instead of after every observation. They control Type I error while
allowing early stopping.

Common boundary types:
- Pocock: Constant boundaries (easier to reject early)
- O'Brien-Fleming: Conservative early, liberal late
- Wang-Tsiatis: Flexible family including both above

References:
    Jennison, C., & Turnbull, B. W. (1999). Group Sequential Methods with Applications to Clinical Trials.
        Chapman & Hall/CRC.
"""

from enum import Enum
from typing import Any

import numpy as np
from pydantic import Field
from scipy import stats

from exploration_sequential_testing.distributions.base import Distribution
from exploration_sequential_testing.distributions.gaussian import (
    GaussianDistribution,
    GaussianParams,
)
from exploration_sequential_testing.tests.base import (
    Decision,
    SequentialTest,
    TestConfig,
)


class BoundaryType(str, Enum):
    """Types of group sequential boundaries."""

    POCOCK = "pocock"
    OBRIEN_FLEMING = "obrien_fleming"
    WANG_TSIATIS = "wang_tsiatis"


class GroupSequentialConfig(TestConfig):
    """Configuration for Group Sequential test.

    Attributes:
        alpha: Type I error rate
        beta: Type II error rate
        n_looks: Number of interim analyses (including final)
        boundary_type: Type of boundary function
        delta: Parameter for Wang-Tsiatis (0=OBF, 0.5=Pocock)
        total_n: Total planned sample size
    """

    n_looks: int = Field(ge=2, description="Number of interim looks")
    boundary_type: BoundaryType = Field(
        default=BoundaryType.OBRIEN_FLEMING, description="Boundary function type"
    )
    delta: float = Field(
        default=0.0, ge=0.0, le=0.5, description="Wang-Tsiatis delta parameter"
    )
    total_n: int = Field(ge=1, description="Total planned sample size")


class GroupSequentialTest(SequentialTest):
    """Group Sequential Test with configurable boundaries.

    This test performs analyses at equally-spaced sample sizes:
    n_1, n_2, ..., n_K where K = n_looks.

    At each look, we compute a Z-statistic and compare to boundaries.
    The boundaries are designed to control the overall Type I error at α.

    Note: Currently implemented for Gaussian distributions (testing means).
    """

    def __init__(
        self,
        config: GroupSequentialConfig,
        null_dist: Distribution,
        alt_dist: Distribution,
    ):
        """Initialize Group Sequential test.

        Args:
            config: Group sequential configuration
            null_dist: Distribution under H0 (must be Gaussian)
            alt_dist: Distribution under H1 (must be Gaussian)

        Raises:
            ValueError: If distributions are not Gaussian
        """
        super().__init__(config)
        self.config: GroupSequentialConfig  # Type hint

        if not isinstance(null_dist, GaussianDistribution) or not isinstance(
            alt_dist, GaussianDistribution
        ):
            raise ValueError(
                "GroupSequentialTest currently only supports Gaussian distributions"
            )

        self.null_dist: GaussianDistribution = null_dist
        self.alt_dist: GaussianDistribution = alt_dist

        # Sample sizes at each look
        self.look_times = np.arange(1, config.n_looks + 1) / config.n_looks
        self.look_ns = (self.look_times * config.total_n).astype(int)

        # Compute boundaries
        self.upper_boundaries, self.lower_boundaries = self._compute_boundaries()

        # State
        self._current_look = 0
        self._sum_x = 0.0
        self._sum_x2 = 0.0

    def reset(self) -> None:
        """Reset test state."""
        super().reset()
        self._current_look = 0
        self._sum_x = 0.0
        self._sum_x2 = 0.0

    def _compute_boundaries(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute decision boundaries for all looks.

        Returns:
            (upper, lower) Z-score boundaries
        """
        K = self.config.n_looks
        t = self.look_times  # Information fractions

        if self.config.boundary_type == BoundaryType.OBRIEN_FLEMING:
            # O'Brien-Fleming: c / sqrt(t)
            # Solve for c to achieve desired alpha
            c = self._solve_for_obf_constant()
            upper = c / np.sqrt(t)
            lower = -upper

        elif self.config.boundary_type == BoundaryType.POCOCK:
            # Pocock: constant boundary
            c = self._solve_for_pocock_constant()
            upper = np.full(K, c)
            lower = -upper

        elif self.config.boundary_type == BoundaryType.WANG_TSIATIS:
            # Wang-Tsiatis: c * t^(-delta)
            delta = self.config.delta
            c = self._solve_for_wt_constant(delta)
            upper = c * t ** (-delta)
            lower = -upper

        else:
            raise ValueError(f"Unknown boundary type: {self.config.boundary_type}")

        return upper, lower

    def _solve_for_obf_constant(self) -> float:
        """Solve for O'Brien-Fleming constant to achieve target alpha."""
        # Approximation: c ≈ Φ^(-1)(1 - α/2) * sqrt(number of looks)
        # This is a rough approximation; exact requires numerical integration
        K = self.config.n_looks
        return stats.norm.ppf(1 - self.config.alpha / 2) * np.sqrt(K) * 0.8

    def _solve_for_pocock_constant(self) -> float:
        """Solve for Pocock constant to achieve target alpha."""
        # Approximation based on number of looks
        K = self.config.n_looks
        # Approximate Pocock constants
        pocock_constants = {
            2: 1.98,
            3: 2.29,
            4: 2.36,
            5: 2.41,
        }
        if K in pocock_constants:
            return pocock_constants[K]
        # General approximation
        return 2.0 + 0.25 * np.log(K)

    def _solve_for_wt_constant(self, delta: float) -> float:
        """Solve for Wang-Tsiatis constant."""
        # Interpolate between OBF (delta=0) and Pocock (delta=0.5)
        if delta == 0.0:
            return self._solve_for_obf_constant()
        elif delta == 0.5:
            return self._solve_for_pocock_constant()
        else:
            # Linear interpolation (rough approximation)
            c_obf = self._solve_for_obf_constant()
            c_pocock = self._solve_for_pocock_constant()
            return c_obf + 2 * delta * (c_pocock - c_obf)

    def update(self, observation: float) -> Decision:
        """Update test with new observation.

        Args:
            observation: New data point

        Returns:
            Decision (only makes decisions at scheduled looks)
        """
        self._n_observations += 1
        self._observations_history.append(observation)
        self._sum_x += observation
        self._sum_x2 += observation**2

        # Check if we've reached a scheduled look
        if self._current_look < len(self.look_ns):
            next_look_n = self.look_ns[self._current_look]

            if self._n_observations >= next_look_n:
                # Perform analysis at this look
                z_stat = self._compute_z_statistic()
                self._test_statistic_history.append(z_stat)

                # Check boundaries
                upper_bound = self.upper_boundaries[self._current_look]
                lower_bound = self.lower_boundaries[self._current_look]

                self._current_look += 1

                if z_stat >= upper_bound:
                    return Decision.REJECT_H0
                elif z_stat <= lower_bound:
                    return Decision.ACCEPT_H0

        return Decision.CONTINUE

    def _compute_z_statistic(self) -> float:
        """Compute Z-statistic for current data.

        Z = (sample_mean - null_mean) / (std / sqrt(n))
        """
        n = self._n_observations
        if n == 0:
            return 0.0

        sample_mean = self._sum_x / n
        null_mean = self.null_dist.params.mean
        std = self.null_dist.params.std

        z = (sample_mean - null_mean) / (std / np.sqrt(n))
        return float(z)

    def get_test_statistic(self) -> float:
        """Get current Z-statistic."""
        return self._compute_z_statistic()

    def get_boundaries(self, n_max: int) -> tuple[np.ndarray, np.ndarray]:
        """Get boundaries as function of sample size.

        Args:
            n_max: Maximum sample size

        Returns:
            (upper, lower) boundaries interpolated over [1, n_max]
        """
        # Interpolate boundaries to all sample sizes
        upper = np.full(n_max, np.nan)
        lower = np.full(n_max, np.nan)

        for i, n in enumerate(self.look_ns):
            if n <= n_max:
                upper[n - 1] = self.upper_boundaries[i]
                lower[n - 1] = self.lower_boundaries[i]

        return upper, lower

    @property
    def name(self) -> str:
        """Return test name."""
        return f"GroupSequential({self.config.boundary_type.value})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize test configuration to dictionary.
        
        Returns:
            Dictionary compatible with TestFactory.create()
        """
        result = {"type": "group_sequential"}
        config_dict = self.config.model_dump()
        # Convert enum to string value
        if "boundary_type" in config_dict:
            config_dict["boundary_type"] = config_dict["boundary_type"].value
        result.update(config_dict)
        return result

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"GroupSequentialTest("
            f"boundary={self.config.boundary_type.value}, "
            f"n_looks={self.config.n_looks}, "
            f"total_n={self.config.total_n})"
        )
