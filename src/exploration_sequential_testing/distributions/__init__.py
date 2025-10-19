"""Distribution module for data generation in sequential testing."""

from exploration_sequential_testing.distributions.base import (
    Distribution,
    DistributionParams,
)
from exploration_sequential_testing.distributions.bernoulli import (
    BernoulliDistribution,
    BernoulliParams,
)
from exploration_sequential_testing.distributions.factory import DistributionFactory
from exploration_sequential_testing.distributions.gaussian import (
    GaussianDistribution,
    GaussianParams,
)

__all__ = [
    "Distribution",
    "DistributionParams",
    "BernoulliDistribution",
    "BernoulliParams",
    "GaussianDistribution",
    "GaussianParams",
    "DistributionFactory",
]
