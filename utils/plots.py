"""
Plots Module for CurrencyVision AI.
Plotly 7.x compatible — uses title=dict(text, font) syntax for axes.
"""

from typing import List, Dict, Tuple
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ── Design Tokens ──────────────────────────────────────────────────────────
_FONT    = "Sora, Inter, sans-serif"
_BG      = "rgba(0,0,0,0)"
_GRID    = "rgba(255,255,255,0.05)"
_TEXT    = "#EDF2FF"
_MUTED   = "#8B9CC8"
_VIOLET  = "#8B5CF6"
_INDIGO  = "#6366F1"
_EMERALD = "#10B981"
_AMBER   = "#F59E0B"
_ROSE    = "#F43F5E"
_TEAL    = "#14B8A6"
_SLATE   = "#334155"


def _axis(title_text: str = "") -> dict:
    """Returns a standard axis dict compatible with Plotly 7.x."""
    d = dict(
        gridcolor=_GRID,
        tickfont=dict(color=_MUTED, size=11),
        zeroline=False,
    )
    if title_text:
        d["title"] = dict(text=title_text, font=dict(color=_MUTED, size=12))
    return d


def _base_layout(height: int = 300, margin: dict = None) -> dict:
    """Returns base Plotly layout dict — Plotly 7.x safe."""
    return dict(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color=_TEXT),
        height=height,
        margin=margin or dict(l=20, r=20, t=50, b=20),
        legend=dict(
            font=dict(color=_TEXT, size=12),
            bgcolor="rgba(13,17,23,0.70)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
        ),
    )


# ─── Confidence Gauge ──────────────────────────────────────────────────────
def plot_confidence_gauge(confidence_pct: float, denomination: str) -> go.Figure:
    """Circular gauge for prediction confidence. Color shifts by score level."""

    bar_color = _EMERALD if confidence_pct >= 80 else (_AMBER if confidence_pct >= 50 else _ROSE)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence_pct,
        number={"suffix": "%", "font": {"size": 44, "color": bar_color, "family": _FONT}},
        title={"text": f"<b>₹{denomination}</b>", "font": {"size": 20, "color": _TEXT, "family": _FONT}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "rgba(255,255,255,0.15)",
                "tickfont": {"color": _MUTED, "size": 10},
            },
            "bar": {"color": bar_color, "thickness": 0.22},
            "bgcolor": "rgba(22,27,39,0.60)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50],   "color": "rgba(244,63,94,0.10)"},
                {"range": [50, 80],  "color": "rgba(245,158,11,0.10)"},
                {"range": [80, 100], "color": "rgba(16,185,129,0.12)"},
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
    """Horizontal bar chart for top class probabilities."""

    sorted_preds = sorted(predictions, key=lambda x: x[1])
    labels  = [f"₹{item[0]}" for item in sorted_preds]
    scores  = [item[1] for item in sorted_preds]
    n       = len(scores)
    colors  = [_VIOLET if i == n - 1 else (_INDIGO if i == n - 2 else _SLATE) for i in range(n)]

    fig = go.Figure(go.Bar(
        x=scores, y=labels, orientation="h",
        text=[f"{s:.1f}%" for s in scores],
        textposition="outside",
        textfont=dict(color=_TEXT, size=12, family=_FONT),
        marker=dict(color=colors, opacity=0.90,
                    line=dict(color="rgba(255,255,255,0.06)", width=1),
                    cornerradius=6),
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
        ("accuracy",     "Train Accuracy", _INDIGO, "solid"),
        ("val_accuracy", "Val Accuracy",   _TEAL,   "dash"),
        ("loss",         "Train Loss",     _ROSE,   "solid"),
        ("val_loss",     "Val Loss",       _AMBER,  "dot"),
    ]

    for key, name, color, dash in cfg:
        if key in history_dict:
            fig.add_trace(go.Scatter(
                x=epochs, y=history_dict[key],
                mode="lines+markers", name=name,
                line=dict(color=color, width=2.5, dash=dash),
                marker=dict(size=6, color=color,
                            line=dict(color="rgba(255,255,255,0.2)", width=1)),
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
    """Interactive confusion matrix heatmap."""

    labels = [f"₹{c}" for c in class_labels]

    fig = px.imshow(
        np.array(cm_matrix),
        x=labels, y=labels,
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
        title=dict(text="<b>Confusion Matrix</b>", font=dict(size=15, color=_TEXT)),
        coloraxis_showscale=False,
        xaxis=dict(tickfont=dict(color=_TEXT, size=12), side="bottom"),
        yaxis=dict(tickfont=dict(color=_TEXT, size=12)),
    )
    fig.update_traces(textfont=dict(color=_TEXT, size=12))
    return fig
