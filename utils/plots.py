"""
Plots Module for CurrencyVision AI.
Provides interactive Plotly visualization components including Circular Confidence Gauge,
Top-3 Probability Bar Chart, Training History Curves, and Confusion Matrix Heatmaps.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def plot_confidence_gauge(confidence_pct: float, denomination: str) -> go.Figure:
    """
    Creates an Apple/OpenAI styled circular gauge indicator for prediction confidence.

    Args:
        confidence_pct (float): Confidence score percentage (0 - 100).
        denomination (str): Predicted currency denomination label.

    Returns:
        go.Figure: Plotly Figure gauge component.
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence_pct,
            number={"suffix": "%", "font": {"size": 42, "color": "#F8FAFC", "family": "Inter, sans-serif"}},
            title={
                "text": f"<b>Confidence: ₹{denomination}</b>",
                "font": {"size": 18, "color": "#94A3B8", "family": "Inter, sans-serif"},
            },
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
                "bar": {"color": "#4F46E5", "thickness": 0.3},
                "bgcolor": "rgba(30, 41, 59, 0.5)",
                "borderwidth": 2,
                "bordercolor": "rgba(255, 255, 255, 0.1)",
                "steps": [
                    {"range": [0, 40], "color": "rgba(239, 68, 68, 0.2)"},
                    {"range": [40, 75], "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [75, 100], "color": "rgba(16, 185, 129, 0.2)"},
                ],
                "threshold": {
                    "line": {"color": "#10B981", "width": 4},
                    "thickness": 0.8,
                    "value": confidence_pct,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(l=30, r=30, t=50, b=20),
    )
    return fig


def plot_top_predictions(predictions: List[Tuple[str, float]]) -> go.Figure:
    """
    Creates a sleek horizontal probability bar chart for Top-3 predictions.

    Args:
        predictions (List[Tuple[str, float]]): List of (class_label, probability_pct) tuples.

    Returns:
        go.Figure: Plotly horizontal bar chart.
    """
    # Sort ascending for bottom-to-top rendering
    sorted_preds = sorted(predictions, key=lambda x: x[1], reverse=False)
    labels = [f"₹{item[0]}" for item in sorted_preds]
    scores = [item[1] for item in sorted_preds]

    colors = ["#6366F1" if i == len(scores) - 1 else "#334155" for i in range(len(scores))]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            text=[f"{s:.1f}%" for s in scores],
            textposition="outside",
            marker=dict(
                color=colors,
                line=dict(color="rgba(255, 255, 255, 0.2)", width=1),
                cornerradius=6,
            ),
        )
    )

    fig.update_layout(
        title=dict(text="<b>Top Predictions Probability</b>", font=dict(size=16, color="#F8FAFC")),
        xaxis=dict(title="Probability (%)", range=[0, 105], gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="", tickfont=dict(size=14, color="#F8FAFC")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(l=20, r=40, t=50, b=20),
    )
    return fig


def plot_training_history(history_dict: Dict[str, List[float]]) -> go.Figure:
    """
    Generates interactive dual-panel line plot showing Training & Validation Accuracy and Loss.

    Args:
        history_dict (Dict[str, List[float]]): Keras model fit history dictionary.

    Returns:
        go.Figure: Plotly Figure with line subplots.
    """
    epochs = list(range(1, len(history_dict.get("accuracy", [])) + 1))

    fig = go.Figure()

    # Accuracy Lines
    if "accuracy" in history_dict:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history_dict["accuracy"],
                mode="lines+markers",
                name="Train Accuracy",
                line=dict(color="#4F46E5", width=3),
            )
        )
    if "val_accuracy" in history_dict:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history_dict["val_accuracy"],
                mode="lines+markers",
                name="Val Accuracy",
                line=dict(color="#10B981", width=3, dash="dash"),
            )
        )

    # Loss Lines
    if "loss" in history_dict:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history_dict["loss"],
                mode="lines+markers",
                name="Train Loss",
                line=dict(color="#EF4444", width=3),
            )
        )
    if "val_loss" in history_dict:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history_dict["val_loss"],
                mode="lines+markers",
                name="Val Loss",
                line=dict(color="#F59E0B", width=3, dash="dash"),
            )
        )

    fig.update_layout(
        title=dict(text="<b>CNN Training Performance History</b>", font=dict(size=18, color="#F8FAFC")),
        xaxis=dict(title="Epochs", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title="Score / Loss", gridcolor="rgba(255,255,255,0.08)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#F8FAFC"), bgcolor="rgba(15, 23, 42, 0.6)"),
        height=380,
        margin=dict(l=30, r=30, t=50, b=30),
    )
    return fig


def plot_confusion_matrix_heatmap(
    cm_matrix: List[List[int]], class_labels: List[str]
) -> go.Figure:
    """
    Creates an interactive Confusion Matrix Heatmap using Plotly.

    Args:
        cm_matrix (List[List[int]]): 2D confusion matrix list/array.
        class_labels (List[str]): List of denomination class labels.

    Returns:
        go.Figure: Plotly Heatmap.
    """
    labels_formatted = [f"₹{c}" for c in class_labels]

    fig = px.imshow(
        np.array(cm_matrix),
        x=labels_formatted,
        y=labels_formatted,
        color_continuous_scale="Viridis",
        labels=dict(x="Predicted Denomination", y="True Denomination", color="Count"),
        text_auto=True,
    )

    fig.update_layout(
        title=dict(text="<b>Confusion Matrix Heatmap</b>", font=dict(size=18, color="#F8FAFC")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC"),
        height=400,
        margin=dict(l=30, r=30, t=50, b=30),
    )
    return fig
