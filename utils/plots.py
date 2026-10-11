"""
Plots Module for CurrencyVision AI.
Provides premium Plotly visualization components with a consistent deep-space dark theme.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ── Shared Design Tokens ────────────────────────────────────────────────────
_FONT = "Sora, Inter, sans-serif"
_PAPER_BG = "rgba(0,0,0,0)"
_PLOT_BG  = "rgba(0,0,0,0)"
_GRID     = "rgba(255,255,255,0.05)"
_TEXT     = "#EDF2FF"
_MUTED    = "#8B9CC8"

_VIOLET   = "#8B5CF6"
_INDIGO   = "#6366F1"
_EMERALD  = "#10B981"
_AMBER    = "#F59E0B"
_ROSE     = "#F43F5E"
_TEAL     = "#14B8A6"
_SLATE    = "#334155"


def _base_layout(**overrides) -> dict:
    """Returns shared Plotly layout dict for consistent theming."""
    base = dict(
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        font=dict(family=_FONT, color=_TEXT),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            font=dict(color=_TEXT, size=12),
            bgcolor="rgba(13,17,23,0.70)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
        ),
    )
    base.update(overrides)
    return base


def plot_confidence_gauge(confidence_pct: float, denomination: str) -> go.Figure:
    """
    Premium circular gauge for prediction confidence score.

    Args:
        confidence_pct (float): Confidence score 0–100.
        denomination (str): Predicted denomination label.

    Returns:
        go.Figure
    """
    # Pick needle/bar color based on confidence level
    if confidence_pct >= 80:
        bar_color = _EMERALD
    elif confidence_pct >= 50:
        bar_color = _AMBER
    else:
        bar_color = _ROSE

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence_pct,
            number={
                "suffix": "%",
                "font": {"size": 44, "color": bar_color, "family": _FONT},
            },
            title={
                "text": f"<b>₹{denomination}</b>",
                "font": {"size": 20, "color": _TEXT, "family": _FONT},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "rgba(255,255,255,0.15)",
                    "tickfont": {"color": _MUTED, "size": 11},
                },
                "bar": {"color": bar_color, "thickness": 0.22},
                "bgcolor": "rgba(22,27,39,0.60)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50],  "color": "rgba(244,63,94,0.10)"},
                    {"range": [50, 80], "color": "rgba(245,158,11,0.10)"},
                    {"range": [80, 100],"color": "rgba(16,185,129,0.12)"},
                ],
                "threshold": {
                    "line": {"color": bar_color, "width": 3},
                    "thickness": 0.75,
                    "value": confidence_pct,
                },
            },
        )
    )
    fig.update_layout(**_base_layout(height=270, margin=dict(l=25, r=25, t=55, b=15)))
    return fig


def plot_top_predictions(predictions: List[Tuple[str, float]]) -> go.Figure:
    """
    Horizontal bar chart for top class probabilities.

    Args:
        predictions: List of (label, probability_pct) tuples.

    Returns:
        go.Figure
    """
    sorted_preds = sorted(predictions, key=lambda x: x[1])
    labels = [f"₹{item[0]}" for item in sorted_preds]
    scores = [item[1] for item in sorted_preds]

    n = len(scores)
    bar_colors = [
        _VIOLET if i == n - 1
        else _INDIGO if i == n - 2
        else _SLATE
        for i in range(n)
    ]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            text=[f"{s:.1f}%" for s in scores],
            textposition="outside",
            textfont=dict(color=_TEXT, size=12, family=_FONT),
            marker=dict(
                color=bar_colors,
                opacity=0.90,
                line=dict(color="rgba(255,255,255,0.06)", width=1),
                cornerradius=6,
            ),
        )
    )

    fig.update_layout(
        **_base_layout(height=270, margin=dict(l=20, r=55, t=50, b=20)),
        title=dict(
            text="<b>Top Predictions</b>",
            font=dict(size=15, color=_TEXT, family=_FONT),
        ),
        xaxis=dict(
            title="Probability (%)",
            range=[0, 115],
            gridcolor=_GRID,
            tickfont=dict(color=_MUTED),
            titlefont=dict(color=_MUTED),
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(size=14, color=_TEXT),
            gridcolor="rgba(0,0,0,0)",
        ),
        bargap=0.35,
    )
    return fig


def plot_training_history(history_dict: Dict[str, List[float]]) -> go.Figure:
    """
    Dual-line chart for training and validation accuracy/loss.

    Args:
        history_dict: Keras-style history dict with accuracy, val_accuracy, loss, val_loss.

    Returns:
        go.Figure
    """
    epochs = list(range(1, len(history_dict.get("accuracy", [])) + 1))

    fig = go.Figure()

    trace_cfg = [
        ("accuracy",     "Train Accuracy", _INDIGO, "solid"),
        ("val_accuracy", "Val Accuracy",   _TEAL,   "dash"),
        ("loss",         "Train Loss",     _ROSE,   "solid"),
        ("val_loss",     "Val Loss",       _AMBER,  "dot"),
    ]

    for key, name, color, dash in trace_cfg:
        if key in history_dict:
            fig.add_trace(go.Scatter(
                x=epochs,
                y=history_dict[key],
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2.5, dash=dash),
                marker=dict(size=6, color=color, line=dict(color="rgba(255,255,255,0.2)", width=1)),
            ))

    fig.update_layout(
        **_base_layout(height=380, margin=dict(l=30, r=30, t=55, b=30)),
        title=dict(
            text="<b>Training Performance</b>",
            font=dict(size=15, color=_TEXT, family=_FONT),
        ),
        xaxis=dict(
            title="Epoch",
            gridcolor=_GRID,
            tickfont=dict(color=_MUTED),
            titlefont=dict(color=_MUTED),
        ),
        yaxis=dict(
            title="Score",
            gridcolor=_GRID,
            tickfont=dict(color=_MUTED),
            titlefont=dict(color=_MUTED),
        ),
        hovermode="x unified",
    )
    return fig


def plot_confusion_matrix_heatmap(
    cm_matrix: List[List[int]], class_labels: List[str]
) -> go.Figure:
    """
    Interactive confusion matrix heatmap.

    Args:
        cm_matrix: 2D confusion matrix.
        class_labels: List of denomination labels.

    Returns:
        go.Figure
    """
    labels_fmt = [f"₹{c}" for c in class_labels]

    fig = px.imshow(
        np.array(cm_matrix),
        x=labels_fmt,
        y=labels_fmt,
        color_continuous_scale=[
            [0.0,  "rgba(22,27,39,1)"],
            [0.25, "rgba(99,102,241,0.4)"],
            [0.60, "rgba(139,92,246,0.75)"],
            [1.0,  "rgba(167,139,250,1)"],
        ],
        labels=dict(x="Predicted", y="Actual", color="Count"),
        text_auto=True,
    )

    fig.update_layout(
        **_base_layout(height=400, margin=dict(l=30, r=30, t=55, b=30)),
        title=dict(
            text="<b>Confusion Matrix</b>",
            font=dict(size=15, color=_TEXT, family=_FONT),
        ),
        coloraxis_showscale=False,
        xaxis=dict(tickfont=dict(color=_TEXT, size=12), side="bottom"),
        yaxis=dict(tickfont=dict(color=_TEXT, size=12)),
    )
    fig.update_traces(textfont=dict(color=_TEXT, size=12))
    return fig
