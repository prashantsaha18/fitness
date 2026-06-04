"""
frontend/components/charts.py
───────────────────────────────
Reusable Plotly chart factories.

Every chart function:
  • Accepts only domain-level arguments (data, labels)
  • Applies the dark theme automatically via theme.plotly_layout()
  • Returns a go.Figure ready for st.plotly_chart()
  • Never calls st.plotly_chart() itself — that belongs in the page

Naming convention: make_<chart_type>()
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from frontend.components.theme import (
    BORDER, CARD_BG, DARK_BG, ELECTRIC, NEON_GREEN, TEXT_DIM, TEXT_MUTED, plotly_layout
)


# ── Heart-rate sparkline ──────────────────────────────────────────────────────

def make_hr_chart(
    hr_values: Sequence[int],
    current_hr: int,
    zone_color: str,
    max_hr: int = 220,
    height: int = 260,
) -> go.Figure:
    """Rolling HR line chart with zone bands and current-HR marker."""
    fig = go.Figure()

    # Zone bands
    for lo, hi, col, lbl in [
        (0,   100, "#10B981", "Resting"),
        (100, 120, "#10B981", "Fat Burn"),
        (120, 150, "#3B82F6", "Cardio"),
        (150, 170, "#F59E0B", "Peak"),
        (170, max_hr + 5, "#EF4444", "Anaerobic"),
    ]:
        fig.add_hrect(
            y0=lo, y1=hi, fillcolor=col, opacity=0.04, line_width=0,
            annotation_text=lbl, annotation_position="right",
            annotation_font=dict(size=9, color=TEXT_MUTED),
        )

    xs = list(range(len(hr_values)))
    fig.add_trace(go.Scatter(
        x=xs, y=list(hr_values),
        mode="lines",
        line=dict(color=ELECTRIC, width=2, shape="spline", smoothing=0.5),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.08)",
        hovertemplate="%{y} bpm<extra></extra>",
    ))

    # Current-value marker
    if hr_values:
        fig.add_trace(go.Scatter(
            x=[xs[-1]], y=[hr_values[-1]], mode="markers",
            marker=dict(size=11, color=zone_color,
                        line=dict(width=2, color="#F9FAFB")),
            showlegend=False,
            hovertemplate=f"Now: {current_hr} bpm<extra></extra>",
        ))

    fig.add_hline(
        y=current_hr, line_dash="dot", line_color=zone_color, opacity=0.5,
        annotation_text=f"{current_hr} bpm",
        annotation_font=dict(color=zone_color, size=10),
    )

    fig.update_layout(
        height=height,
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(range=[40, max_hr + 10], title="bpm"),
        **{k: v for k, v in plotly_layout().items() if k not in ("xaxis", "yaxis")},
    )
    fig.update_layout(margin=dict(l=0, r=55, t=10, b=0))
    return fig


# ── Gauge ─────────────────────────────────────────────────────────────────────

def make_gauge(
    value_0_to_1: float,
    title: str,
    bar_color: str,
    height: int = 180,
) -> go.Figure:
    """Circular gauge for a normalised 0-1 metric (rendered as 0-100%)."""
    pct = round(value_0_to_1 * 100, 1)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number=dict(suffix="%", font=dict(size=26, color="#F9FAFB", family="JetBrains Mono")),
        title=dict(text=title, font=dict(size=11, color=TEXT_MUTED)),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color=TEXT_MUTED, size=9),
                      tickcolor=BORDER),
            bar=dict(color=bar_color, thickness=0.8),
            bgcolor=CARD_BG, bordercolor=BORDER,
            steps=[
                dict(range=[0,  33], color=CARD_BG),
                dict(range=[33, 66], color="#0D0D1A"),
                dict(range=[66, 100], color=DARK_BG),
            ],
        ),
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor=DARK_BG,
        font_color="#F9FAFB",
    )
    return fig


# ── Radar / spider chart ──────────────────────────────────────────────────────

def make_radar(
    categories: list[str],
    values: list[float],
    fill_color: str = ELECTRIC,
    height: int = 260,
) -> go.Figure:
    """Radar chart for multi-dimensional user profile."""
    # Close the polygon
    cats = categories + [categories[0]]
    vals = values + [values[0]]

    r, g, b = int(fill_color[1:3], 16), int(fill_color[3:5], 16), int(fill_color[5:7], 16)

    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats,
        fill="toself",
        fillcolor=f"rgba({r},{g},{b},0.12)",
        line=dict(color=fill_color, width=2),
        marker=dict(size=6, color=fill_color),
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=40, r=40, t=20, b=20),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        polar=dict(
            bgcolor=CARD_BG,
            radialaxis=dict(visible=True, range=[0, 100],
                            color=TEXT_MUTED, gridcolor=BORDER,
                            tickfont=dict(size=9)),
            angularaxis=dict(color=TEXT_DIM, tickfont=dict(size=11)),
        ),
        showlegend=False,
    )
    return fig


# ── Bar chart ─────────────────────────────────────────────────────────────────

def make_bar(
    x: list[Any],
    y: list[float],
    colors: list[str] | str = ELECTRIC,
    title_x: str = "",
    title_y: str = "",
    height: int = 240,
    text: list[str] | None = None,
) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=x, y=y,
        marker_color=colors,
        text=text,
        textposition="outside" if text else "none",
        textfont=dict(color=TEXT_DIM, size=11),
        hovertemplate="%{x}: %{y}<extra></extra>",
    ))
    layout = plotly_layout()
    layout.update(
        height=height,
        xaxis=dict(gridcolor=BORDER, color=TEXT_MUTED, title=title_x),
        yaxis=dict(gridcolor=BORDER, color=TEXT_MUTED, title=title_y),
        showlegend=False,
    )
    fig.update_layout(**layout)
    return fig


# ── Line / area chart ─────────────────────────────────────────────────────────

def make_line(
    x: list[Any],
    y_series: dict[str, list[float]],
    colors: dict[str, str] | None = None,
    fill: bool = False,
    height: int = 240,
    title_x: str = "",
    title_y: str = "",
) -> go.Figure:
    colors = colors or {}
    fig = go.Figure()
    for i, (name, vals) in enumerate(y_series.items()):
        color = colors.get(name, [ELECTRIC, NEON_GREEN, "#8B5CF6", "#FF6B35"][i % 4])
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig.add_trace(go.Scatter(
            x=x, y=vals, name=name,
            mode="lines+markers",
            line=dict(color=color, width=2, shape="spline", smoothing=0.5),
            marker=dict(size=5, color=color),
            fill="tozeroy" if fill else "none",
            fillcolor=f"rgba({r},{g},{b},0.10)" if fill else None,
            hovertemplate=f"{name}: %{{y}}<extra></extra>",
        ))
    layout = plotly_layout()
    layout.update(
        height=height,
        xaxis=dict(gridcolor=BORDER, color=TEXT_MUTED, title=title_x),
        yaxis=dict(gridcolor=BORDER, color=TEXT_MUTED, title=title_y),
    )
    fig.update_layout(**layout)
    return fig


# ── Training history (dual-axis) ──────────────────────────────────────────────

def make_training_history(
    epochs: list[int],
    val_auc: list[float],
    train_loss: list[float],
    best_epoch: int | None = None,
    height: int = 260,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=epochs, y=val_auc, name="Val AUC",
        mode="lines+markers",
        line=dict(color=NEON_GREEN, width=2),
        marker=dict(size=6),
        hovertemplate="Epoch %{x} — AUC %{y:.4f}<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=epochs, y=train_loss, name="Train Loss",
        mode="lines+markers",
        line=dict(color="#FF6B35", width=2, dash="dot"),
        marker=dict(size=5),
        hovertemplate="Epoch %{x} — Loss %{y:.4f}<extra></extra>",
    ), secondary_y=True)

    if best_epoch is not None:
        fig.add_vline(
            x=best_epoch, line_dash="dash", line_color="#8B5CF6",
            annotation_text="Best checkpoint",
            annotation_font=dict(color="#8B5CF6", size=10),
        )

    layout = plotly_layout()
    layout.update(height=height)
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Val AUC",    gridcolor=BORDER, secondary_y=False)
    fig.update_yaxes(title_text="Train Loss", gridcolor=BORDER, secondary_y=True)
    fig.update_xaxes(title_text="Epoch",      gridcolor=BORDER)
    return fig


# ── Correlation heatmap ───────────────────────────────────────────────────────

def make_heatmap(
    z: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    height: int = 360,
) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=z, x=x_labels, y=y_labels,
        colorscale=[[0, "#1A0A2E"], [0.5, CARD_BG], [1.0, ELECTRIC]],
        zmid=0,
        text=np.round(z, 2),
        texttemplate="%{text}",
        textfont=dict(size=10, color="#F9FAFB"),
        hovertemplate="%{x} × %{y}: %{z:.3f}<extra></extra>",
    ))
    layout = plotly_layout()
    layout.update(
        height=height,
        xaxis=dict(side="bottom", tickangle=-30, gridcolor=BORDER),
        yaxis=dict(gridcolor=BORDER),
        coloraxis_showscale=False,
    )
    fig.update_layout(**layout)
    return fig


# ── Posterior Beta distribution (A/B testing) ─────────────────────────────────

def make_posterior(
    alpha_a: float, beta_a: float,
    alpha_b: float, beta_b: float,
    rate_a: float, rate_b: float,
    height: int = 280,
) -> go.Figure:
    from scipy.stats import beta as beta_dist

    x    = np.linspace(0, 1, 500)
    y_a  = beta_dist.pdf(x, alpha_a, beta_a)
    y_b  = beta_dist.pdf(x, alpha_b, beta_b)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y_a, name="Control A", fill="tozeroy",
        line=dict(color="#8B5CF6", width=2),
        fillcolor="rgba(139,92,246,0.14)",
        hovertemplate="Rate %{x:.2%}: density %{y:.2f}<extra>A</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y_b, name="Treatment B", fill="tozeroy",
        line=dict(color=NEON_GREEN, width=2),
        fillcolor="rgba(57,255,20,0.10)",
        hovertemplate="Rate %{x:.2%}: density %{y:.2f}<extra>B</extra>",
    ))
    fig.add_vline(x=rate_a, line_dash="dot", line_color="#8B5CF6", opacity=0.7,
                   annotation_text=f"A {rate_a:.1%}",
                   annotation_font=dict(color="#8B5CF6", size=10))
    fig.add_vline(x=rate_b, line_dash="dot", line_color=NEON_GREEN, opacity=0.7,
                   annotation_text=f"B {rate_b:.1%}",
                   annotation_font=dict(color=NEON_GREEN, size=10))

    layout = plotly_layout()
    layout.update(
        height=height,
        xaxis=dict(gridcolor=BORDER, title="Completion Rate", tickformat=".0%"),
        yaxis=dict(gridcolor=BORDER, title="Probability Density"),
    )
    fig.update_layout(**layout)
    return fig
