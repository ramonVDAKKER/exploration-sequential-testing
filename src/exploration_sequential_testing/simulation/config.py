"""Configuration for Monte Carlo simulations."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from exploration_sequential_testing.distributions.base import Distribution
from exploration_sequential_testing.tests.base import SequentialTest


class Hypothesis(str, Enum):
    """Which hypothesis data is generated under."""

    NULL = "null"
    ALTERNATIVE = "alternative"


class SimulationConfig(BaseModel):
    """Configuration for Monte Carlo simulation.

    Attributes:
        n_simulations: Number of Monte Carlo replications
        max_n: Maximum sample size before forced stopping
        test: Sequential test to apply
        null_dist: Distribution under null hypothesis
        alt_dist: Distribution under alternative hypothesis
        hypothesis: Which hypothesis to generate data under
        seed: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        batch_size: Number of simulations per batch (for progress tracking)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    n_simulations: int = Field(ge=100, description="Number of Monte Carlo replications")
    max_n: int = Field(ge=25, description="Maximum sample size per trajectory")
    test: SequentialTest = Field(description="Sequential test instance")
    null_dist: Distribution = Field(description="Null hypothesis distribution")
    alt_dist: Distribution = Field(description="Alternative hypothesis distribution")
    hypothesis: Hypothesis = Field(
        default=Hypothesis.NULL, description="Data generating hypothesis"
    )
    seed: int | None = Field(default=None, description="Random seed")
    n_jobs: int = Field(default=-1, description="Number of parallel jobs")
    batch_size: int = Field(default=100, ge=1, description="Simulations per batch")

    def get_data_dist(self) -> Distribution:
        """Get the distribution to generate data from.

        Returns:
            Distribution based on which hypothesis is being simulated
        """
        return self.null_dist if self.hypothesis == Hypothesis.NULL else self.alt_dist
