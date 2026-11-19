"""Godown inventory helpers: group pipes by diameter, draw size-coded results."""
from __future__ import annotations

import re

import altair as alt
import cv2
import numpy as np
import pandas as pd

PALETTE = [
    "#0F6B6B", "#E8B923", "#C2410C", "#2563EB", "#7C3AED",
    "#16A34A", "#DB2777", "#0891B2", "#65A30D", "#9A3412",
]
OTHER = "Other size"
UNCERTAIN = "Uncertain (partial)"
NOT_CALIBRATED = "Not calibrated"
SPECIAL_COLORS = {OTHER: "#6B7280", UNCERTAIN: "#9CA3AF", NOT_CALIBRATED: "#9CA3AF"}


def parse_sizes(text: str) -> list[float]:
    """Parse '20, 25, 32' into [20.0, 25.0, 32.0] (sorted, unique, positive)."""
    nums = {float(n) for n in re.findall(r"\d+(?:\.\d+)?", text or "") if float(n) > 0}
    return sorted(nums)


def build_color_map(standards: list[float]) -> dict[str, str]:
    cmap = {f"{s:g} mm": PALETTE[i % len(PALETTE)] for i, s in enumerate(standards)}
    cmap.update(SPECIAL_COLORS)
    return cmap


def diameter_px(w: int, h: int) -> tuple[float, float]:
    """Return (diameter in px, aspect ratio min/max).

    Roughly round boxes use the mean side; oval (tilted or cut-off) boxes use
    the longer side, which is closer to the true diameter.
    """
    aspect = min(w, h) / max(w, h, 1)
    return (((w + h) / 2) if aspect >= 0.85 else max(w, h)), aspect


def assign_size(d_mm: float | None, aspect: float, standards: list[float],
                tol_pct: float, min_aspect: float) -> str:
    if d_mm is None:
        return NOT_CALIBRATED
    if aspect < min_aspect:
        return UNCERTAIN
    if not standards:
        return f"{round(d_mm):g} mm"
    nearest = min(standards, key=lambda s: abs(s - d_mm))
    if abs(nearest - d_mm) / nearest <= tol_pct / 100:
        return f"{nearest:g} mm"
    return OTHER


def build_table(boxes: list[dict], px_per_mm: float | None, standards: list[float],
                tol_pct: float, min_aspect: float) -> pd.DataFrame:
    rows = []
    for b in boxes:
        w, h = b["x2"] - b["x1"], b["y2"] - b["y1"]
        d_px, aspect = diameter_px(w, h)
        d_mm = round(d_px / px_per_mm, 2) if px_per_mm else None
        rows.append({
            "ID": b["ID"],
            "Confidence": b["Confidence"],
            "Diameter (mm)": d_mm,
            "Size": assign_size(d_mm, aspect, standards, tol_pct, min_aspect),
            "Roundness": round(aspect, 2),
            "d_px": d_px,
            "x1": b["x1"], "y1": b["y1"], "x2": b["x2"], "y2": b["y2"],
        })
    return pd.DataFrame(rows)


def _hex_to_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def draw_inventory(image_bgr: np.ndarray, df: pd.DataFrame, cmap: dict[str, str]) -> np.ndarray:
    """Draw each pipe as a filled, size-colored ellipse with its ID number."""
    out = image_bgr.copy()
    if df.empty:
        return out
    scale = max(out.shape[:2]) / 1000
    thick = max(1, int(round(2 * scale)))
    fscale = max(0.4, 0.55 * scale)

    overlay = out.copy()
    for r in df.itertuples():
        center = ((r.x1 + r.x2) // 2, (r.y1 + r.y2) // 2)
        axes = (max((r.x2 - r.x1) // 2, 1), max((r.y2 - r.y1) // 2, 1))
        cv2.ellipse(overlay, center, axes, 0, 0, 360, _hex_to_bgr(cmap.get(r.Size, "#6B7280")), -1)
    out = cv2.addWeighted(overlay, 0.40, out, 0.60, 0)

    for r in df.itertuples():
        center = ((r.x1 + r.x2) // 2, (r.y1 + r.y2) // 2)
        axes = (max((r.x2 - r.x1) // 2, 1), max((r.y2 - r.y1) // 2, 1))
        cv2.ellipse(out, center, axes, 0, 0, 360, _hex_to_bgr(cmap.get(r.Size, "#6B7280")), thick, cv2.LINE_AA)
        text = str(r.ID)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fscale, 1)
        pos = (center[0] - tw // 2, center[1] + th // 2)
        cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, fscale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, fscale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def summarize(df: pd.DataFrame, cmap: dict[str, str]) -> pd.DataFrame:
    """Pipes per size, in the order of the size list, with share and mean diameter."""
    if df.empty:
        return pd.DataFrame(columns=["Size", "Pipes", "Share", "Avg diameter (mm)"])
    grouped = df.groupby("Size").agg(Pipes=("ID", "count"), avg=("Diameter (mm)", "mean"))
    order = [s for s in cmap if s in grouped.index] + [s for s in grouped.index if s not in cmap]
    grouped = grouped.loc[order].reset_index()
    grouped["Share"] = (grouped["Pipes"] / grouped["Pipes"].sum() * 100).round(1).astype(str) + "%"
    grouped["Avg diameter (mm)"] = grouped["avg"].round(2)
    return grouped[["Size", "Pipes", "Share", "Avg diameter (mm)"]]


def donut_chart(summary: pd.DataFrame, cmap: dict[str, str]) -> alt.Chart:
    domain = [s for s in summary["Size"]]
    return (
        alt.Chart(summary)
        .mark_arc(innerRadius=55)
        .encode(
            theta=alt.Theta("Pipes:Q"),
            color=alt.Color("Size:N", legend=None,
                            scale=alt.Scale(domain=domain, range=[cmap.get(s, "#6B7280") for s in domain])),
            tooltip=["Size", "Pipes", "Share"],
        )
        .properties(height=220)
    )


def chip_html(label: str, count: int, hex_color: str) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    text = "#14232B" if (0.299 * r + 0.587 * g + 0.114 * b) > 160 else "#FFFFFF"
    return (f'<span style="display:inline-block;background:{hex_color};color:{text};'
            f'padding:4px 12px;border-radius:6px;margin:2px 6px 2px 0;font-weight:500">'
            f'{label}: {count}</span>')
