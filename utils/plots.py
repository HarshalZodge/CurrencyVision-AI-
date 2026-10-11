"""
Plots Module for CurrencyVision AI.
Light Aesthetic theme — crisp typography, high contrast, Plotly 7.x compatible.
"""

from typing import List, Dict, Tuple
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ── Design Tokens for Light Aesthetic ──────────────────────────────────────
_FONT    = "Sora, Inter, -apple-system, sans-serif"
_BG      = "rgba(0,0,0,0)"
_GRID    = "rgba(15, 23, 42, 0.06)"
_TEXT    = "#0F172A"       # Deep charcoal for headings/labels
_MUTED   = "#64748B"       # Refined slate for subtitles/axis marks
_INDIGO  = "#4F46E5"       # Primary vibrant indigo
_VIOLET  = "#7C3AED"       # Accent violet
_EMERALD = "#059669"       # Success green
_AMBER   = "#D97706"       # Warning amber
_ROSE    = "#E11D48"       # Error / low confidence rose
_SLATE_LIGHT = "#E2E8F0"   # Subtle bar fill


def _base_layout(height: int = 300, margin: dict = None) -> dict:
    """Returns base Plotly layout dict with light aesthetic styling."""
    return dict(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color=_TEXT),
        height=height,
        margin=margin or dict(l=20, r=20, t=50, b=20),
        legend=dict(
            font=dict(color=_TEXT, size=12),
            bgcolor="rgba(255, 255, 255, 0.90)",
            bordercolor="rgba(226, 232, 240, 0.9)",
            borderwidth=1,
        ),
    )


# ─── Confidence Gauge ──────────────────────────────────────────────────────
def plot_confidence_gauge(confidence_pct: float, denomination: str) -> go.Figure:
    """Circular gauge for prediction confidence score in light aesthetic."""

    bar_color = _EMERALD if confidence_pct >= 80 else (_AMBER if confidence_pct >= 50 else _ROSE)

    fig = go.Figure(go.Indicator(
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
                "tickcolor": "rgba(15, 23, 42, 0.2)",
                "tickfont": {"color": _MUTED, "size": 10},
            },
            "bar": {"color": bar_color, "thickness": 0.22},
            "bgcolor": "rgba(241, 245, 249, 0.95)",
            "borderwidth": 1,
            "bordercolor": "rgba(226, 232, 240, 0.9)",
            "steps": [
                {"range": [0, 50],   "color": "rgba(254, 226, 226, 0.6)"},
                {"range": [50, 80],  "color": "rgba(254, 243, 199, 0.6)"},
                {"range": [80, 100], "color": "rgba(209, 250, 229, 0.6)"},
            ],
            "threshold": {
                "line": {"color": bar_color, "width": 3},
                "thickness": 0.75,
                "value": confidence_pct,
            },
        },
    ))

    fig.update_layout(**_base_layout(height=270, margin=dict(l=25, r=25, t=55, b=15)))
    return fig


# ─── Top Predictions Bar Chart ─────────────────────────────────────────────
def plot_top_predictions(predictions: List[Tuple[str, float]]) -> go.Figure:
    """Horizontal bar chart for top class probabilities in light aesthetic."""

    sorted_preds = sorted(predictions, key=lambda x: x[1])
    labels  = [f"₹{item[0]}" for item in sorted_preds]
    scores  = [item[1] for item in sorted_preds]
    n       = len(scores)

    colors  = [
        _INDIGO if i == n - 1 else ("#818CF8" if i == n - 2 else _SLATE_LIGHT)
        for i in range(n)
    ]

    fig = go.Figure(go.Bar(
        x=scores, y=labels, orientation="h",
        text=[f"{s:.1f}%" for s in scores],
        textposition="outside",
        textfont=dict(color=_TEXT, size=12, family=_FONT),
        marker=dict(
            color=colors,
            line=dict(color="rgba(15, 23, 42, 0.08)", width=1),
            cornerradius=6,
        ),
    ))

    fig.update_layout(
        **_base_layout(height=270, margin=dict(l=20, r=55, t=50, b=20)),
        title=dict(text="<b>Top Predictions</b>", font=dict(size=15, color=_TEXT)),
        xaxis=dict(
            title=dict(text="Probability (%)", font=dict(color=_MUTED, size=12)),
            range=[0, 115],
            gridcolor=_GRID,
            tickfont=dict(color=_MUTED),
            zeroline=False,
        ),
        yaxis=dict(tickfont=dict(size=14, color=_TEXT), gridcolor=_BG),
        bargap=0.35,
    )
    return fig


# ─── Training History Lines ─────────────────────────────────────────────────
def plot_training_history(history_dict: Dict[str, List[float]]) -> go.Figure:
    """Dual-line chart for training and validation accuracy/loss."""

    epochs = list(range(1, len(history_dict.get("accuracy", [])) + 1))
    fig    = go.Figure()

    cfg = [
        ("accuracy",     "Train Accuracy", _INDIGO,  "solid"),
        ("val_accuracy", "Val Accuracy",   _EMERALD, "dash"),
        ("loss",         "Train Loss",     _ROSE,    "solid"),
        ("val_loss",     "Val Loss",       _AMBER,   "dot"),
    ]

    for key, name, color, dash in cfg:
        if key in history_dict:
            fig.add_trace(go.Scatter(
                x=epochs, y=history_dict[key],
                mode="lines+markers", name=name,
                line=dict(color=color, width=2.5, dash=dash),
                marker=dict(size=6, color=color,
                            line=dict(color="#FFFFFF", width=1.5)),
            ))

    fig.update_layout(
        **_base_layout(height=380, margin=dict(l=30, r=30, t=55, b=30)),
        title=dict(text="<b>Training Performance</b>", font=dict(size=15, color=_TEXT)),
        xaxis=dict(
            title=dict(text="Epoch", font=dict(color=_MUTED, size=12)),
            gridcolor=_GRID,
            tickfont=dict(color=_MUTED),
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text="Score", font=dict(color=_MUTED, size=12)),
            gridcolor=_GRID,
            tickfont=dict(color=_MUTED),
            zeroline=False,
        ),
        hovermode="x unified",
    )
    return fig


# ─── Confusion Matrix Heatmap ───────────────────────────────────────────────
def plot_confusion_matrix_heatmap(
    cm_matrix: List[List[int]], class_labels: List[str]
) -> go.Figure:
    """Interactive confusion matrix heatmap styled for light aesthetic."""

    labels = [f"₹{c}" for c in class_labels]

    fig = px.imshow(
        np.array(cm_matrix),
        x=labels, y=labels,
        color_continuous_scale=[
            [0.0,  "#F8FAFC"],
            [0.2,  "#EEF2FF"],
            [0.5,  "#C7D2FE"],
            [0.8,  "#818CF8"],
            [1.0,  "#4F46E5"],
        ],
        labels=dict(x="Predicted", y="Actual", color="Count"),
        text_auto=True,
    )

    fig.update_layout(
        **_base_layout(height=400, margin=dict(l=30, r=30, t=55, b=30)),
        title=dict(text="<b>Confusion Matrix</b>", font=dict(size=15, color=_TEXT)),
        coloraxis_showscale=False,
        xaxis=dict(tickfont=dict(color=_TEXT, size=12), side="bottom"),
        yaxis=dict(tickfont=dict(color=_TEXT, size=12)),
    )
    fig.update_traces(textfont=dict(color=_TEXT, size=12))
    return fig
