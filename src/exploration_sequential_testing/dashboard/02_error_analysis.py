"""Error analysis page: Type I and Type II errors vs stopping time."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import numpy as np

st.title(":material/monitoring: Error Analysis")

if "null_results" not in st.session_state and "alt_results" not in st.session_state:
    st.warning(""":material/warning: No results available. Please run a Monte Carlo study first.
               Navigate to **Configuration** to execute a study.""")
    st.stop()

null_results = st.session_state.get("null_results")
alt_results = st.session_state.get("alt_results")
metrics = st.session_state.get("metrics")

# Display metrics if available
if metrics is not None:
    st.header(":material/monitoring: Error Rate Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        type_i = metrics.error_metrics.type_i_error
        alpha = 0.05  # Default, try to get from config
        if "test_config" in st.session_state:
            alpha = st.session_state["test_config"].get("alpha", 0.05)

        delta = ((type_i - alpha) / alpha * 100) if alpha > 0 else 0
        st.metric(
            "Type I Error",
            f"{type_i:.4f}",
            delta=f"{delta:+.1f}% vs α",
            delta_color="inverse",
        )
        if type_i > alpha:
            st.caption(":material/warning: Exceeds nominal α")
        else:
            st.caption(":material/check: Within nominal α")

    with col2:
        if metrics.error_metrics.type_ii_error is not None:
            type_ii = metrics.error_metrics.type_ii_error
            beta = 0.20  # Default
            if "test_config" in st.session_state:
                beta = st.session_state["test_config"].get("beta", 0.20)

            delta = ((type_ii - beta) / beta * 100) if beta > 0 else 0
            st.metric(
                "Type II Error",
                f"{type_ii:.4f}",
                delta=f"{delta:+.1f}% vs β",
                delta_color="inverse",
            )
            if type_ii > beta:
                st.caption(":material/warning: Exceeds nominal β")
            else:
                st.caption(":material/check: Within nominal β")

    with col3:
        if metrics.error_metrics.power is not None:
            power = metrics.error_metrics.power
            target_power = 1 - 0.20  # Default
            if "test_config" in st.session_state:
                target_power = 1 - st.session_state["test_config"].get("beta", 0.20)

            delta = power - target_power
            st.metric(
                "Power",
                f"{power:.4f}",
                delta=f"{delta:+.4f}",
                delta_color="normal",
            )

    with col4:
        if null_results is not None:
            prob_continue = metrics.error_metrics.prob_continue_under_null
            st.metric(
                "P(Continue | H₀)",
                f"{prob_continue:.4f}",
            )
            st.caption("Truncated trajectories")

st.header(":material/monitoring: Error Rates vs Stopping Time")

tabs = st.tabs(["Type I Error", "Type II Error", "Combined View"])

# Type I Error tab
with tabs[0]:
    if null_results is not None:
        st.subheader("Type I Error by Stopping Time")
        st.markdown("""
        This plot shows the **Type I error rate** (probability of false positive)
        as a function of when a decision was made.

        - Each point represents trajectories that stopped at time *n*
        - The horizontal line shows the nominal α level
        """)

        df = null_results.raw_results

        # Group by stopping time and compute error rate
        error_by_time = (
            df.group_by("stopping_time")
            .agg(
                [
                    pl.count().alias("n_trajectories"),
                    (pl.col("decision") == "reject_h0")
                    .sum()
                    .alias("n_errors"),
                ]
            )
            .sort("stopping_time")
        )

        error_by_time = error_by_time.with_columns(
            (pl.col("n_errors") / pl.col("n_trajectories")).alias("error_rate")
        )

        # Create plot
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=error_by_time["stopping_time"],
                y=error_by_time["error_rate"],
                mode="markers+lines",
                name="Type I Error",
                marker=dict(
                    size=error_by_time["n_trajectories"] / error_by_time["n_trajectories"].max() * 20 + 5,
                    color=error_by_time["error_rate"],
                    colorscale="Reds",
                    showscale=True,
                    colorbar=dict(title="Error Rate"),
                ),
                text=error_by_time["n_trajectories"],
                hovertemplate="<b>Stopping Time:</b> %{x}<br>"
                + "<b>Type I Error:</b> %{y:.4f}<br>"
                + "<b>N Trajectories:</b> %{text}<br>"
                + "<extra></extra>",
            )
        )

        # Add nominal alpha line
        if "test_config" in st.session_state:
            alpha = st.session_state["test_config"].get("alpha", 0.05)
            fig.add_hline(
                y=alpha,
                line_dash="dash",
                line_color="red",
                annotation_text=f"α = {alpha:.3f}",
            )

        fig.update_layout(
            title="Type I Error Rate by Stopping Time",
            xaxis_title="Stopping Time (n)",
            yaxis_title="Type I Error Rate",
            hovermode="closest",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Histogram of stopping times colored by decision
        st.subheader("Stopping Time Distribution (Under H₀)")

        fig2 = px.histogram(
            df.to_pandas(),
            x="stopping_time",
            color="decision",
            barmode="stack",
            title="Distribution of Stopping Times by Decision",
            labels={"stopping_time": "Stopping Time", "count": "Frequency"},
            color_discrete_map={
                "reject_h0": "red",
                "accept_h0": "green",
                "continue": "gray",
            },
        )

        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("No results under H₀. Run simulation to see Type I error analysis.")

# Type II Error tab
with tabs[1]:
    if alt_results is not None:
        st.subheader("Type II Error by Stopping Time")
        st.markdown("""
        This plot shows the **Type II error rate** (probability of false negative)
        as a function of when a decision was made.

        - Each point represents trajectories that stopped at time *n*
        - The horizontal line shows the nominal β level
        """)

        df = alt_results.raw_results

        # Group by stopping time and compute error rate
        error_by_time = (
            df.group_by("stopping_time")
            .agg(
                [
                    pl.count().alias("n_trajectories"),
                    (pl.col("decision") == "accept_h0")
                    .sum()
                    .alias("n_errors"),
                ]
            )
            .sort("stopping_time")
        )

        error_by_time = error_by_time.with_columns(
            (pl.col("n_errors") / pl.col("n_trajectories")).alias("error_rate")
        )

        # Create plot
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=error_by_time["stopping_time"],
                y=error_by_time["error_rate"],
                mode="markers+lines",
                name="Type II Error",
                marker=dict(
                    size=error_by_time["n_trajectories"] / error_by_time["n_trajectories"].max() * 20 + 5,
                    color=error_by_time["error_rate"],
                    colorscale="Blues",
                    showscale=True,
                    colorbar=dict(title="Error Rate"),
                ),
                text=error_by_time["n_trajectories"],
                hovertemplate="<b>Stopping Time:</b> %{x}<br>"
                + "<b>Type II Error:</b> %{y:.4f}<br>"
                + "<b>N Trajectories:</b> %{text}<br>"
                + "<extra></extra>",
            )
        )

        # Add nominal beta line
        if "test_config" in st.session_state:
            beta = st.session_state["test_config"].get("beta", 0.20)
            fig.add_hline(
                y=beta,
                line_dash="dash",
                line_color="blue",
                annotation_text=f"β = {beta:.3f}",
            )

        fig.update_layout(
            title="Type II Error Rate by Stopping Time",
            xaxis_title="Stopping Time (n)",
            yaxis_title="Type II Error Rate",
            hovermode="closest",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Histogram of stopping times colored by decision
        st.subheader("Stopping Time Distribution (Under H₁)")

        fig2 = px.histogram(
            df.to_pandas(),
            x="stopping_time",
            color="decision",
            barmode="stack",
            title="Distribution of Stopping Times by Decision",
            labels={"stopping_time": "Stopping Time", "count": "Frequency"},
            color_discrete_map={
                "reject_h0": "green",
                "accept_h0": "red",
                "continue": "gray",
            },
        )

        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("No results under H₁. Run simulation to see Type II error analysis.")

# Combined view
with tabs[2]:
    if null_results is not None and alt_results is not None:
        st.subheader("Combined Error Rate Comparison")

        col_a, col_b = st.columns(2)

        with col_a:
            # Box plot of stopping times by hypothesis
            df_null = null_results.raw_results.with_columns(
                pl.lit("H₀").alias("hypothesis")
            )
            df_alt = alt_results.raw_results.with_columns(
                pl.lit("H₁").alias("hypothesis")
            )
            df_combined = pl.concat([df_null, df_alt])

            fig = px.box(
                df_combined.to_pandas(),
                x="hypothesis",
                y="stopping_time",
                color="hypothesis",
                title="Stopping Time by Hypothesis",
                labels={"stopping_time": "Stopping Time", "hypothesis": "Hypothesis"},
            )

            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            # Violin plot
            fig = px.violin(
                df_combined.to_pandas(),
                x="hypothesis",
                y="stopping_time",
                color="hypothesis",
                box=True,
                title="Stopping Time Distribution",
                labels={"stopping_time": "Stopping Time", "hypothesis": "Hypothesis"},
            )

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Run simulations under both hypotheses to see combined view.")
