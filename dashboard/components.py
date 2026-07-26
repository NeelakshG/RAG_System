"""HTML/CSS rendering helpers for the Streamlit dashboard.

Streamlit has no native primitive for a track+fill meter or a bordered card
with the spacing this project wants, so these render plain HTML through
st.markdown(unsafe_allow_html=True). Colors and specs (meter track = lighter
step of the fill's own ramp, status color always paired with an icon+label,
categorical hues assigned in a fixed order) follow the project's data-viz
palette rather than being picked by eye.
"""

import html

STYLE = """
<style>
:root {
  --surface-1: #fcfcfb;
  --surface-2: #f9f9f7;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #898781;
  --border: rgba(11,11,11,0.10);
  --meter-track: #cde2fb;
  --meter-fill: #2a78d6;
  --status-good: #0ca30c;
  --status-critical: #d03b3b;
  --chip-fixed: #2a78d6;
  --chip-recursive: #1baf7a;
  --chip-semantic: #eda100;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface-1: #1a1a19;
    --surface-2: #0d0d0d;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #898781;
    --border: rgba(255,255,255,0.10);
    --meter-track: #184f95;
    --meter-fill: #3987e5;
    --status-good: #0ca30c;
    --status-critical: #e66767;
    --chip-fixed: #3987e5;
    --chip-recursive: #199e70;
    --chip-semantic: #c98500;
  }
}

.hero-answer {
  font-size: 1.3rem;
  line-height: 1.55;
  font-weight: 500;
  color: var(--text-primary);
  padding: 20px 24px;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--surface-1);
  margin-bottom: 18px;
}
.hero-answer.fallback {
  color: var(--text-secondary);
  font-style: italic;
  border-color: var(--status-critical);
}

.section-label {
  font-size: 0.75rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin: 22px 0 10px 0;
  font-weight: 600;
}

.meter { margin-bottom: 12px; }
.meter-label {
  display: flex; justify-content: space-between; align-items: baseline;
  font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 4px;
}
.meter-value { font-variant-numeric: tabular-nums; color: var(--text-primary); font-weight: 600; }
.meter-track { height: 8px; border-radius: 4px; background: var(--meter-track); overflow: hidden; }
.meter-fill { height: 100%; border-radius: 4px; }

.badge {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 0.78rem; font-weight: 600; padding: 2px 9px; border-radius: 100px;
  border: 1px solid var(--border);
}
.badge.supported { color: var(--status-good); }
.badge.unsupported { color: var(--status-critical); }

.chunk-card {
  border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px;
  margin-bottom: 10px; background: var(--surface-1);
}
.chunk-card-head {
  display: flex; justify-content: space-between; align-items: baseline;
  font-size: 0.8rem; color: var(--text-muted); margin-bottom: 6px;
}
.chunk-card-source { color: var(--text-secondary); font-weight: 600; }
.chunk-card-text { font-size: 0.92rem; color: var(--text-primary); line-height: 1.5; }

.chip {
  display: inline-block; width: 9px; height: 9px; border-radius: 50%;
  margin-right: 6px; vertical-align: middle;
}

.source-chip {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 0.78rem; font-weight: 600; color: var(--text-secondary);
  padding: 2px 9px; margin: 0 6px 6px 0; border-radius: 100px;
  border: 1px solid var(--border); background: var(--surface-2);
}
.citation-source {
  color: var(--text-muted); font-weight: 400;
}
</style>
"""

STRATEGY_COLORS = {
    "fixed": "var(--chip-fixed)",
    "recursive": "var(--chip-recursive)",
    "semantic": "var(--chip-semantic)",
}


def render_meter(label: str, value: float | None, status_aware: bool = False, threshold: float = 0.5) -> str:
    """A track+fill meter. status_aware colors the fill by whether value
    clears threshold (used for the composite score, which literally gates
    the fallback) -- the three sub-scores stay on the plain sequential
    ramp since they're diagnostic, not a pass/fail signal on their own.
    """
    if value is None:
        pct, value_text, fill_color = 0, "n/a", "var(--text-muted)"
    else:
        pct = max(0, min(100, round(value * 100)))
        value_text = f"{value:.2f}"
        fill_color = "var(--meter-fill)"
        if status_aware:
            fill_color = "var(--status-good)" if value >= threshold else "var(--status-critical)"
    return (
        f'<div class="meter"><div class="meter-label"><span>{html.escape(label)}</span>'
        f'<span class="meter-value">{value_text}</span></div>'
        f'<div class="meter-track"><div class="meter-fill" style="width:{pct}%; background:{fill_color};"></div></div></div>'
    )


def render_badge(supported: bool) -> str:
    cls = "supported" if supported else "unsupported"
    icon = "✓" if supported else "✗"
    label = "Supported" if supported else "Unsupported"
    return f'<span class="badge {cls}">{icon} {label}</span>'


def render_chunk_card(source_name: str, section_heading: str | None, score: float, text: str, rank: int) -> str:
    heading = f" &middot; {html.escape(section_heading)}" if section_heading else ""
    preview = text if len(text) <= 400 else text[:400].rstrip() + "…"
    return (
        '<div class="chunk-card"><div class="chunk-card-head">'
        f'<span>#{rank} <span class="chunk-card-source">{html.escape(source_name)}</span>{heading}</span>'
        f'<span>score {score:.3f}</span></div>'
        f'<div class="chunk-card-text">{html.escape(preview)}</div></div>'
    )


def render_sources(source_names: list[str]) -> str:
    """One chip per distinct source document behind this answer, in the
    order chunks were retrieved (i.e. most-relevant document first).
    """
    chips = "".join(f'<span class="source-chip">📄 {html.escape(name)}</span>' for name in source_names)
    return f'<div style="margin-bottom:16px;">{chips}</div>'


def render_strategy_chip(strategy: str) -> str:
    color = STRATEGY_COLORS.get(strategy, "var(--text-muted)")
    return f'<span class="chip" style="background:{color};"></span>{html.escape(strategy)}'
