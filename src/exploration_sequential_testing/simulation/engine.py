"""Monte Carlo simulation engine for sequential tests."""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from exploration_sequential_testing.simulation.config import (
    Hypothesis,
    SimulationConfig,
)
from exploration_sequential_testing.simulation.results import SimulationResults
from exploration_sequential_testing.tests.base import Decision, TestResult


def _run_single_trajectory(
    test_state: dict,
    dist_state: dict,
    max_n: int,
    seed: int,
) -> TestResult:
    """Run a single test trajectory (for parallel execution).

    This function is designed to be pickled for multiprocessing.

    Args:
        test_state: Serialized test configuration
        dist_state: Serialized distribution configuration
        max_n: Maximum sample size
        seed: Random seed for this trajectory

    Returns:
        TestResult
    """
    from exploration_sequential_testing.distributions.factory import DistributionFactory
    from exploration_sequential_testing.tests.factory import TestFactory

    # Reconstruct objects
    null_dist = DistributionFactory.create(dist_state["null"])
    alt_dist = DistributionFactory.create(dist_state["alt"])
    test = TestFactory.create(test_state, null_dist, alt_dist)

    # Get data distribution
    hypothesis = dist_state["hypothesis"]
    data_dist = null_dist if hypothesis == "null" else alt_dist

    # Create RNG
    rng = np.random.default_rng(seed)

    # Reset test
    test.reset()

    # Run trajectory
    for _ in range(max_n):
        obs = data_dist.sample(1, rng)[0]
        decision = test.update(obs)

        if decision != Decision.CONTINUE:
            return test.get_result(decision)

    # Truncated at max_n
    return test.get_result(Decision.CONTINUE)


class MonteCarloEngine:
    """Engine for running Monte Carlo simulations for sequential tests.

    This class handles:
    - Running simulations in parallel
    - Progress tracking
    - Result aggregation

    Example:
        >>> engine = MonteCarloEngine()
        >>> config = SimulationConfig(...)
        >>> results = engine.run(config, verbose=True)
    """

    def __init__(self):
        """Initialize Monte Carlo engine."""
        pass

    def run(
        self,
        config: SimulationConfig,
        verbose: bool = True,
    ) -> SimulationResults:
        """Run Monte Carlo simulation.

        Args:
            config: Simulation configuration
            verbose: Whether to show progress bar

        Returns:
            SimulationResults with all trajectory results
        """
        # Serialize objects for multiprocessing
        test_state = config.test.to_dict()
        dist_state = {
            "null": config.null_dist.to_dict(),
            "alt": config.alt_dist.to_dict(),
            "hypothesis": config.hypothesis.value,
        }

        # Generate seeds for each simulation
        if config.seed is not None:
            rng = np.random.default_rng(config.seed)
            seeds = rng.integers(0, 2**31, size=config.n_simulations)
        else:
            seeds = np.random.randint(0, 2**31, size=config.n_simulations)

        # Determine number of workers
        n_jobs = config.n_jobs
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        n_jobs = max(1, min(n_jobs, mp.cpu_count()))

        # Run simulations in parallel
        test_results = []

        if n_jobs == 1:
            # Serial execution
            sim_iterator = range(config.n_simulations)
            if verbose:
                sim_iterator = tqdm(sim_iterator, desc="Running simulations")

            for i in sim_iterator:
                result = _run_single_trajectory(
                    test_state, dist_state, config.max_n, int(seeds[i])
                )
                test_results.append(result)

        else:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = [
                    executor.submit(
                        _run_single_trajectory,
                        test_state,
                        dist_state,
                        config.max_n,
                        int(seeds[i]),
                    )
                    for i in range(config.n_simulations)
                ]

                # Collect results with progress bar
                future_iterator = as_completed(futures)
                if verbose:
                    future_iterator = tqdm(
                        future_iterator, total=config.n_simulations, desc="Running simulations"
                    )

                for future in future_iterator:
                    result = future.result()
                    test_results.append(result)

        # Package results
        return SimulationResults.from_test_results(
            test_results=test_results,
            hypothesis=config.hypothesis.value,
            config_dict=config.model_dump(exclude={"test", "null_dist", "alt_dist"}),
        )

    def run_under_null(
        self,
        config: SimulationConfig,
        verbose: bool = True,
    ) -> SimulationResults:
        """Run simulation under null hypothesis (for Type I error).

        Args:
            config: Simulation configuration
            verbose: Whether to show progress bar

        Returns:
            SimulationResults under H0
        """
        config.hypothesis = Hypothesis.NULL
        return self.run(config, verbose=verbose)

    def run_under_alternative(
        self,
        config: SimulationConfig,
        verbose: bool = True,
    ) -> SimulationResults:
        """Run simulation under alternative hypothesis (for Type II error).

        Args:
            config: Simulation configuration
            verbose: Whether to show progress bar

        Returns:
            SimulationResults under H1
        """
        config.hypothesis = Hypothesis.ALTERNATIVE
        return self.run(config, verbose=verbose)

    def run_both_hypotheses(
        self,
        config: SimulationConfig,
        verbose: bool = True,
    ) -> tuple[SimulationResults, SimulationResults]:
        """Run simulations under both null and alternative hypotheses.

        Args:
            config: Simulation configuration
            verbose: Whether to show progress bar

        Returns:
            Tuple of (null_results, alt_results)
        """
        if verbose:
            print("Running simulations under null hypothesis...")
        null_results = self.run_under_null(config, verbose=verbose)

        if verbose:
            print("\nRunning simulations under alternative hypothesis...")
        alt_results = self.run_under_alternative(config, verbose=verbose)

        return null_results, alt_results
