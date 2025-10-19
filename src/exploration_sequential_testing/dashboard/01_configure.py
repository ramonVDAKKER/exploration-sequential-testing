"""Configuration and execution page for Monte Carlo simulations."""

import streamlit as st

from exploration_sequential_testing.distributions.factory import DistributionFactory
from exploration_sequential_testing.tests.factory import TestFactory
from exploration_sequential_testing.simulation.engine import MonteCarloEngine
from exploration_sequential_testing.simulation.config import SimulationConfig, Hypothesis
from exploration_sequential_testing.metrics.calculator import MetricsCalculator

st.markdown("### :material/settings: Configure & run Monte Carlo simulations")

if "config_history" not in st.session_state:
    st.session_state["config_history"] = []

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### :material/counter_1: Distribution Configuration")

    dist_type = st.selectbox(
        "Distribution",
        ["bernoulli", "gaussian"],
        help="Type of data distribution",
    )

    st.markdown("##### Null Hypothesis (H₀)")
    if dist_type == "gaussian":
        null_mean = st.number_input("Mean (μ₀)", value=0.0, step=0.1, key="null_mean")
        null_std = st.number_input(
            "Std Dev (σ)", value=1.0, min_value=0.01, step=0.1, key="null_std"
        )
        null_config = {"type": "gaussian", "mean": null_mean, "std": null_std}
    elif dist_type == "bernoulli":
        null_p = st.slider(
            "Success Probability (p₀)",
            min_value=0.001,
            max_value=0.999,
            value=0.5,
            step=0.01,
            key="null_p",
        )
        null_config = {"type": "bernoulli", "p": null_p}
    else:
        st.error("Unsupported distribution type")
        st.stop()

    st.markdown("##### Alternative Hypothesis (H₁)")
    if dist_type == "gaussian":
        alt_mean = st.number_input("Mean (μ₁)", value=0.5, step=0.1, key="alt_mean")
        alt_std = st.number_input(
            "Std Dev (σ)", value=1.0, min_value=0.01, step=0.1, key="alt_std"
        )
        alt_config = {"type": "gaussian", "mean": alt_mean, "std": alt_std}
    elif dist_type == "bernoulli":
        alt_p = st.slider(
            "Success Probability (p₁)",
            min_value=0.001,
            max_value=0.999,
            value=0.7,
            step=0.01,
            key="alt_p",
        )
        alt_config = {"type": "bernoulli", "p": alt_p}

        odds_ratio = (alt_p / (1 - alt_p)) / (null_p / (1 - null_p))
        st.info(f"**Odds Ratio:** {odds_ratio:.3f}")

with col2:
    st.markdown("#### :material/counter_2: Test Configuration")

    test_type = st.selectbox(
        "Sequential Test",
        ["sprt"],
        format_func=lambda x: {
            "sprt": "SPRT (Wald's Sequential Probability Ratio Test)",
        }[x],
    )

    alpha = st.slider(
        "Type I Error Rate (α)",
        min_value=0.001,
        max_value=0.10,
        value=0.05,
        step=0.01,
        format="%.3f",
        help="Significance level",
    )

    beta = st.slider(
        "Type II Error Rate (β)",
        min_value=0.001,
        max_value=0.50,
        value=0.20,
        step=0.01,
        format="%.2f",
        help="1 - Power",
    )

    st.info(f"**Power:** {1 - beta:.2f}")

    # Test-specific parameters
    if test_type == "sprt":
        use_truncation = st.checkbox("Enable truncation", value=True)
        if use_truncation:
                    truncate_at = st.number_input(
            "Truncate at sample size",
            min_value=10,
            max_value=10000,
            value=1000,
            step=100,
            help="Maximum sample size (None for no truncation)",
        )

        test_config = {
            "type": "sprt",
            "alpha": alpha,
            "beta": beta,
            "truncate_at": truncate_at if use_truncation else None,
        }

    else:  # group_sequential
        if dist_type != "gaussian":
            st.error("Group Sequential test currently only supports Gaussian distributions")
            st.stop()

        n_looks = st.slider(
            "Number of Interim Looks",
            min_value=2,
            max_value=10,
            value=5,
            help="Number of interim analyses",
        )

        boundary_type = st.selectbox(
            "Boundary Type",
            ["obrien_fleming", "pocock"],
            format_func=lambda x: {
                "obrien_fleming": "O'Brien-Fleming",
                "pocock": "Pocock",
            }[x],
        )

        total_n = st.number_input(
            "Total Sample Size",
            min_value=10,
            max_value=10000,
            value=200,
            step=10,
        )

        test_config = {
            "type": "group_sequential",
            "alpha": alpha,
            "beta": beta,
            "n_looks": n_looks,
            "boundary_type": boundary_type,
            "total_n": total_n,
        }

st.divider()

st.markdown("#### :material/counter_3: Monte Carlo study settings")

col3, col4 = st.columns(2)

with col3:
    n_simulations = st.number_input(
        "Number of Monte Carlo Replications",
        min_value=1_000,
        max_value=1_000_000,
        value=25_000,
        step=5_000,
        help="More replications = more accurate",
    )

    max_n = st.number_input(
        "Maximum Sample Size per Trajectory",
        min_value=50,
        max_value=10_000,
        value=250 if test_type == "sprt" else total_n,
        step=100,
    )

with col4:
    seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=999999,
        value=45,
        help="For reproducibility",
    )

    n_jobs = st.slider(
        "Parallel Jobs",
        min_value=1,
        max_value=16,
        value=-1,
        help="-1 uses all available CPUs",
    )

st.divider()

# Run button
st.markdown("#### :material/counter_4: Run Monte Carlo Simulations")

run_both = st.checkbox(
    "Run under both hypotheses",
    value=True,
    help="Runs simulations under both H₀ (for Type I error) and H₁ (for Type II error)",
)

if st.button(":material/directions_run: Run Simulation", type="primary", use_container_width=True):
    try:
        null_dist = DistributionFactory.create(null_config)
        alt_dist = DistributionFactory.create(alt_config)

        test = TestFactory.create(test_config, null_dist, alt_dist)

        with st.expander(":material/summarize: Configuration Summary", expanded=False):
            st.json(
                {
                    "distribution": {
                        "null": null_config,
                        "alternative": alt_config,
                    },
                    "test": test_config,
                    "simulation": {
                        "n_simulations": n_simulations,
                        "max_n": max_n,
                        "seed": seed,
                        "n_jobs": n_jobs,
                    },
                }
            )

        # Run simulations
        engine = MonteCarloEngine()

        if run_both:
            with st.spinner("Running simulations under both H₀ and H₁..."):

                # Create configurations
                null_sim_config = SimulationConfig(
                    n_simulations=n_simulations,
                    max_n=max_n,
                    test=test,
                    null_dist=null_dist,
                    alt_dist=alt_dist,
                    hypothesis=Hypothesis.NULL,
                    seed=seed,
                    n_jobs=n_jobs,
                )

                alt_sim_config = SimulationConfig(
                    n_simulations=n_simulations,
                    max_n=max_n,
                    test=test,
                    null_dist=null_dist,
                    alt_dist=alt_dist,
                    hypothesis=Hypothesis.ALTERNATIVE,
                    seed=seed + 1,  # Different seed
                    n_jobs=n_jobs,
                )

                # Run under null
                with st.status("Running under H₀...", expanded=True) as status:
                    null_results = engine.run(null_sim_config, verbose=False)
                    status.update(label=":material/check: Completed H₀", state="complete")

                # Run under alternative
                with st.status("Running under H₁...", expanded=True) as status:
                    alt_results = engine.run(alt_sim_config, verbose=False)
                    status.update(label=":material/check: Completed H₁", state="complete")

                # Store in session state
                st.session_state["null_results"] = null_results
                st.session_state["alt_results"] = alt_results
                st.session_state["test_config"] = test_config
                st.session_state["dist_config"] = {"null": null_config, "alt": alt_config}

                # Compute metrics
                if dist_type == "gaussian":
                    metrics = MetricsCalculator.compute_full_metrics(
                        null_results=null_results,
                        alt_results=alt_results,
                        alpha=alpha,
                        beta=beta,
                        effect_size=effect_size,
                    )
                else:
                    metrics = MetricsCalculator.compute_full_metrics(
                        null_results=null_results,
                        alt_results=alt_results,
                        alpha=alpha,
                        beta=beta,
                    )

                st.session_state["metrics"] = metrics

                st.success(":material/check: Simulation completed!")
                st.info("Navigate to other pages to explore detailed results!")

        else:
            hypothesis = st.radio(
                "Generate data under:",
                [Hypothesis.NULL, Hypothesis.ALTERNATIVE],
                format_func=lambda x: "H₀ (Null)" if x == Hypothesis.NULL else "H₁ (Alternative)",
            )

            sim_config = SimulationConfig(
                n_simulations=n_simulations,
                max_n=max_n,
                test=test,
                null_dist=null_dist,
                alt_dist=alt_dist,
                hypothesis=hypothesis,
                seed=seed,
                n_jobs=n_jobs,
            )

            with st.status("Running simulation...", expanded=True) as status:
                results = engine.run(sim_config, verbose=False)
                status.update(label=":material/check: Completed", state="complete")

            if hypothesis == Hypothesis.NULL:
                st.session_state["null_results"] = results
            else:
                st.session_state["alt_results"] = results

            st.success(":material/check: Simulation completed!")
            st.info(f"Results stored for hypothesis: {hypothesis.value}")

    except Exception as e:
        st.error(f":material/error: Error during simulation: {str(e)}")
        st.exception(e)
