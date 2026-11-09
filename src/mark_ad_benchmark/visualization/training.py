import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mark_ad_benchmark.logging import ComponentLogger


def visualize_training(run_dir: Path, logger: ComponentLogger) -> Path | None:
    """Render saved training diagnostics without loading or fitting a model."""
    history_path = run_dir / "training_history.json"
    if not history_path.is_file():
        return None
    history = json.loads(history_path.read_text())
    plots = (
        ("estimated_entropy", "Estimated entropy · fixed monitoring pairs"),
        ("projected_gradient_norm", "Monitoring tangent-gradient norm"),
        ("subspace_change", "One-update subspace change"),
        ("bandwidth_squared", "Adaptive bandwidth σ²"),
    )
    figure = make_subplots(
        rows=len(plots),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[title for _, title in plots],
    )
    for row, (key, title) in enumerate(plots, start=1):
        figure.add_trace(
            go.Scatter(
                x=[point["iteration"] for point in history],
                y=[point[key] for point in history],
                mode="lines+markers",
                name=title,
            ),
            row=row,
            col=1,
        )
    figure.update_xaxes(title_text="SGD update", row=len(plots), col=1)
    figure.update_layout(
        title={
            "text": "Training diagnostics<br><sup>Adaptive bandwidth and stochastic updates do not imply monotonic entropy or convergence.</sup>"
        },
        height=1100,
        showlegend=False,
        template="plotly_white",
        hovermode="x unified",
        margin={"t": 120},
    )
    output_path = run_dir / "training.html"
    figure.write_html(output_path, include_plotlyjs=True, config={"displaylogo": False})
    logger.info(f"Wrote interactive training diagnostics: {output_path}")
    return output_path
