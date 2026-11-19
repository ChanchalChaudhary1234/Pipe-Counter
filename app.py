"""Pipe Counter - warehouse pipe detection UI (Streamlit + YOLO).

Run with:  streamlit run app.py
"""
from __future__ import annotations

import tempfile
import time
from io import BytesIO
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from utils.analysis import size_group
from utils.detector import (
    MODELS_DIR,
    bgr_to_rgb,
    decode_image,
    detect_boxes,
    encode_png,
    list_local_models,
    load_model,
    run_detection,
)

from utils.inventory import (
    build_color_map,
    build_table,
    chip_html,
    diameter_px,
    donut_chart,
    draw_inventory,
    parse_sizes,
    summarize,
)

# ----------------------------------------------------------------------------
# Page setup and styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Pipe Counter",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

    .block-container { padding-top: 1.6rem; max-width: 1350px; }

    .hero {
        background: #14232B;
        color: #F3F5F4;
        border-radius: 14px;
        padding: 26px 32px;
        margin-bottom: 22px;
        border-bottom: 6px solid #E8B923;
    }
    .hero h1 { margin: 0 0 4px 0; font-size: 2rem; font-weight: 700; color: #F3F5F4; }
    .hero p  { margin: 0; font-size: 1rem; color: #B9C7C4; }

    .stat {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 14px 18px;
        border-left: 5px solid #0F6B6B;
        box-shadow: 0 1px 3px rgba(20, 35, 43, 0.12);
    }
    .stat.accent { border-left-color: #E8B923; }
    .stat .label { font-size: 0.85rem; color: #5B6B70; margin-bottom: 2px; }
    .stat .value { font-size: 1.9rem; font-weight: 700; color: #14232B; line-height: 1.15; }

    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background: #E4EAE8; border-radius: 8px 8px 0 0; padding: 8px 18px; font-weight: 500;
    }
    .stTabs [aria-selected="true"] { background: #0F6B6B !important; color: #FFFFFF !important; }

    section[data-testid="stSidebar"] { border-right: 1px solid #D3DCD9; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Pipe Counter</h1>
        <p>Count, size and sort the pipes in your warehouse from a photo, a camera or a video.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []


def stat_card(label: str, value, accent: bool = False) -> str:
    cls = "stat accent" if accent else "stat"
    return f'<div class="{cls}"><div class="label">{label}</div><div class="value">{value}</div></div>'


# ----------------------------------------------------------------------------
# Sidebar: model + settings
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Model")

    local_models = list_local_models()
    uploaded_model = st.file_uploader("Upload a trained model (.pt)", type=["pt"])

    model_path: str | None = None
    if uploaded_model is not None:
        MODELS_DIR.mkdir(exist_ok=True)
        saved = MODELS_DIR / uploaded_model.name
        saved.write_bytes(uploaded_model.getbuffer())
        model_path = str(saved)
        st.success(f"Using uploaded model: {uploaded_model.name}")
    elif local_models:
        chosen = st.selectbox("Model in models/ folder", local_models)
        model_path = str(MODELS_DIR / chosen)
    else:
        st.warning(
            "No trained model found. Put your `best.pt` in the `models/` folder, "
            "or upload it above. Until then the demo model `yolov8n.pt` is used, "
            "which does not know pipes."
        )
        model_path = "yolov8n.pt"

    st.header("Detection")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.40, 0.05,
                     help="Raise it to remove false detections, lower it to find more pipes.")
    iou = st.slider("Overlap threshold (IoU)", 0.10, 0.95, 0.50, 0.05,
                    help="Lower it to merge overlapping boxes in dense bundles.")
    imgsz = st.select_slider("Image size", options=[320, 480, 640, 800, 1024, 1280], value=640,
                             help="Larger sizes find small pipes in dense bundles but run slower.")

    st.header("Size calibration")
    use_calibration = st.toggle("Measure diameter in mm", value=False)
    px_per_mm = None
    if use_calibration:
        px_per_mm = st.number_input(
            "Pixels per millimetre", min_value=0.1, value=10.0, step=0.5,
            help="Photograph a ruler or a known object at the same distance as the pipes. "
                 "Pixels per mm = object width in pixels / real width in mm.",
        )
        with st.expander("Calculate from a reference object"):
            ref_px = st.number_input("Object width in pixels", min_value=1.0, value=500.0)
            ref_mm = st.number_input("Object width in mm", min_value=0.1, value=50.0)
            st.caption(f"Result: {ref_px / ref_mm:.2f} px/mm. Enter it above.")

    model = load_model(model_path)

@st.cache_data(show_spinner=False, max_entries=30)
def cached_boxes(_model, model_path: str, img_bytes: bytes, conf: float, iou: float, imgsz: int):
    """Detect once per photo and settings, so moving other widgets does not re-run the model."""
    return detect_boxes(_model, decode_image(img_bytes), conf, iou, imgsz)


# ----------------------------------------------------------------------------
# Shared result renderer
# ----------------------------------------------------------------------------


def show_results(image_bgr, source_name: str, key: str) -> None:
    """Run detection on one image and render the full result panel."""
    with st.spinner("Detecting pipes..."):
        started = time.perf_counter()
        annotated, detections = run_detection(model, image_bgr, conf, iou, imgsz, px_per_mm)
        elapsed = time.perf_counter() - started

    df = pd.DataFrame(detections)
    total = len(df)
    avg_conf = f"{df['Confidence'].mean():.0%}" if total else "-"
    n_colors = df["Color"].nunique() if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(stat_card("Pipes counted", total, accent=True), unsafe_allow_html=True)
    c2.markdown(stat_card("Average confidence", avg_conf), unsafe_allow_html=True)
    c3.markdown(stat_card("Colors found", n_colors), unsafe_allow_html=True)
    c4.markdown(stat_card("Processing time", f"{elapsed * 1000:.0f} ms"), unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.image(bgr_to_rgb(annotated), caption=f"{source_name} - numbered detections",
                 use_container_width=True)

    with right:
        if total == 0:
            st.info("No pipes found. Try lowering the confidence threshold in the sidebar.")
        else:
            by_color = df["Color"].value_counts().rename_axis("Color").reset_index(name="Pipes")
            st.subheader("Pipes by color")
            st.bar_chart(by_color, x="Color", y="Pipes", color="#0F6B6B", height=220)

            if px_per_mm:
                df["Size group"] = df["Diameter (mm)"].apply(size_group)
                by_size = df["Size group"].value_counts().sort_index().rename_axis("Size").reset_index(name="Pipes")
                st.subheader("Pipes by diameter")
                st.bar_chart(by_size, x="Size", y="Pipes", color="#E8B923", height=220)
            else:
                st.caption("Turn on size calibration in the sidebar to see counts by diameter.")

    if total:
        st.subheader("Detection details")
        table_cols = [c for c in df.columns if c not in ("x1", "y1", "x2", "y2")]
        st.dataframe(df[table_cols], use_container_width=True, hide_index=True)

        d1, d2 = st.columns(2)
        d1.download_button("Download annotated image", encode_png(annotated),
                           file_name=f"pipes_{datetime.now():%Y%m%d_%H%M%S}.png",
                           mime="image/png", key=f"png_{key}", use_container_width=True)
        d2.download_button("Download results (CSV)", df[table_cols].to_csv(index=False).encode(),
                           file_name=f"pipes_{datetime.now():%Y%m%d_%H%M%S}.csv",
                           mime="text/csv", key=f"csv_{key}", use_container_width=True)

    # save a history entry (once per distinct source/time)
    entry_id = f"{key}-{source_name}-{total}-{conf}-{iou}-{imgsz}"
    if not st.session_state.history or st.session_state.history[-1]["id"] != entry_id:
        st.session_state.history.append(
            {
                "id": entry_id,
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Source": source_name,
                "Pipes": total,
                "Avg confidence": avg_conf,
            }
        )


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_inv, tab_image, tab_snap, tab_live, tab_video, tab_history = st.tabs(
    ["Godown inventory", "Upload images", "Camera snapshot", "Live camera", "Video file", "History"]
)

# --- Godown inventory -------------------------------------------------------
with tab_inv:
    st.subheader("Count the godown by pipe diameter")
    st.caption(
        "Add one photo per rack or bay, taken straight on to the pipe ends. Each photo is counted, "
        "sorted into your pipe sizes, and added to the godown total."
    )

    ic1, ic2, ic3 = st.columns(3)
    sizes_text = ic1.text_input(
        "Pipe sizes in your godown (mm)", "20, 25, 32, 40, 50",
        help="List the diameters your godown stocks, separated by commas. Each pipe is matched to the closest one.",
    )
    tol_pct = ic2.slider("Size match tolerance (%)", 5, 30, 15,
                         help="How far a measured diameter can be from a listed size and still match it.")
    min_aspect = ic3.slider("Roundness needed to measure size", 0.3, 1.0, 0.7, 0.05,
                            help="Pipes that look squashed are usually cut off by the photo edge or seen at a steep angle. "
                                 "They are counted but marked uncertain instead of guessing a size.")
    standards = parse_sizes(sizes_text)
    cmap = build_color_map(standards)

    photos = st.file_uploader(
        "Add photos (one per rack or bay)", type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True, key="inv_files",
    )

    if not photos:
        st.info("Add photos to start the count. Turn on 'Measure diameter in mm' in the sidebar, "
                "or calibrate each photo with a pipe of known size.")
    else:
        summary_slot = st.container()
        all_tables = []

        for idx, f in enumerate(photos):
            img_bytes = f.getvalue()
            img = decode_image(img_bytes)
            if img is None:
                st.error(f"Could not read {f.name}. Use a JPG, PNG, BMP or WEBP file.")
                continue

            with st.spinner(f"Counting pipes in {f.name}..."):
                boxes = cached_boxes(model, model_path, img_bytes, conf, iou, imgsz)

            st.markdown(f"### {f.name}")

            photo_px_per_mm = px_per_mm
            with st.expander("Calibrate this photo with a pipe of known size (recommended)"):
                st.caption("Find one pipe whose real diameter you know, read its number on the picture below, "
                           "and enter both. All other pipes in this photo are then measured against it.")
                k1, k2 = st.columns(2)
                ref_id = k1.number_input("Reference pipe number (0 = off)", min_value=0, max_value=max(len(boxes), 0),
                                         value=0, step=1, key=f"refid{idx}")
                ref_mm = k2.number_input("Its real diameter (mm)", min_value=0.0, value=0.0, step=0.5, key=f"refmm{idx}")
                if ref_id > 0 and ref_mm > 0:
                    rb = boxes[int(ref_id) - 1]
                    ref_px, _ = diameter_px(rb["x2"] - rb["x1"], rb["y2"] - rb["y1"])
                    photo_px_per_mm = ref_px / ref_mm
                    st.success(f"Calibrated: {photo_px_per_mm:.2f} pixels per mm for this photo.")

            table = build_table(boxes, photo_px_per_mm, standards, tol_pct, min_aspect)
            annotated = draw_inventory(img, table, cmap)
            summary = summarize(table, cmap)

            m1, m2, m3 = st.columns(3)
            m1.markdown(stat_card("Pipes in this photo", len(table), accent=True), unsafe_allow_html=True)
            top = summary.sort_values("Pipes", ascending=False).iloc[0]["Size"] if len(summary) else "-"
            m2.markdown(stat_card("Most common size", top), unsafe_allow_html=True)
            m3.markdown(stat_card("Sizes found", summary["Size"].nunique() if len(summary) else 0),
                        unsafe_allow_html=True)
            st.write("")

            left, right = st.columns([3, 2], gap="large")
            with left:
                st.image(bgr_to_rgb(annotated), use_container_width=True,
                         caption="Each pipe is colored by size. Numbers match the table.")
                if len(summary):
                    st.markdown(" ".join(chip_html(r.Size, int(r.Pipes), cmap.get(r.Size, "#6B7280"))
                                         for r in summary.itertuples()), unsafe_allow_html=True)
            with right:
                if len(summary):
                    st.altair_chart(donut_chart(summary, cmap), use_container_width=True)
                    st.dataframe(summary, hide_index=True, use_container_width=True)
                else:
                    st.info("No pipes found. Try lowering the confidence threshold in the sidebar.")

            if photo_px_per_mm is None and len(table):
                st.warning("Sizes are not measured yet. Turn on 'Measure diameter in mm' in the sidebar or "
                           "calibrate this photo above.")

            if len(table):
                table = table.assign(Photo=f.name)
                all_tables.append(table)
            st.divider()

        # ---- godown total, shown at the top of the tab ----
        with summary_slot:
            if all_tables:
                full = pd.concat(all_tables, ignore_index=True)
                total_summary = summarize(full, cmap)
                st.markdown("### Godown total")
                t1, t2, t3 = st.columns(3)
                t1.markdown(stat_card("Total pipes", len(full), accent=True), unsafe_allow_html=True)
                t2.markdown(stat_card("Photos counted", len(all_tables)), unsafe_allow_html=True)
                t3.markdown(stat_card("Sizes found", total_summary["Size"].nunique()), unsafe_allow_html=True)
                st.write("")

                a, b = st.columns([2, 3], gap="large")
                with a:
                    st.dataframe(total_summary, hide_index=True, use_container_width=True)
                with b:
                    by_photo = full.pivot_table(index="Photo", columns="Size", values="ID",
                                                aggfunc="count", fill_value=0)
                    ordered = [c for c in cmap if c in by_photo.columns]
                    st.bar_chart(by_photo[ordered + [c for c in by_photo.columns if c not in ordered]],
                                 color=[cmap.get(c, "#6B7280") for c in
                                        ordered + [c for c in by_photo.columns if c not in ordered]],
                                 height=240)
                    st.caption("Pipes per photo, split by size.")

                export_cols = ["Photo", "ID", "Size", "Diameter (mm)", "Confidence", "Roundness"]
                buf = BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as xl:
                    total_summary.to_excel(xl, sheet_name="Godown total", index=False)
                    by_photo.reset_index().to_excel(xl, sheet_name="By photo", index=False)
                    full[export_cols].to_excel(xl, sheet_name="All pipes", index=False)
                e1, e2 = st.columns(2)
                e1.download_button("Download inventory (Excel)", buf.getvalue(),
                                   file_name=f"godown_inventory_{datetime.now():%Y%m%d_%H%M}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)
                e2.download_button("Download all pipes (CSV)", full[export_cols].to_csv(index=False).encode(),
                                   file_name=f"godown_pipes_{datetime.now():%Y%m%d_%H%M}.csv",
                                   mime="text/csv", use_container_width=True)
                st.divider()

# --- Upload images ----------------------------------------------------------
with tab_image:
    files = st.file_uploader(
        "Drop one or more photos of your pipes",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )
    if not files:
        st.info("Upload a photo to count the pipes in it. Photos taken straight on, "
                "facing the pipe ends, give the best results.")
    for idx, f in enumerate(files or []):
        img = decode_image(f.getvalue())
        if img is None:
            st.error(f"Could not read {f.name}. Use a JPG, PNG, BMP or WEBP file.")
            continue
        st.markdown(f"### {f.name}")
        show_results(img, f.name, key=f"upload{idx}")
        st.divider()

# --- Camera snapshot --------------------------------------------------------
with tab_snap:
    st.caption("Take a photo with your device camera. Your browser will ask for camera access.")
    shot = st.camera_input("Take a photo of the pipes")
    if shot is not None:
        img = decode_image(shot.getvalue())
        if img is not None:
            show_results(img, "Camera snapshot", key="snap")

# --- Live camera ------------------------------------------------------------
with tab_live:
    st.caption("Streams from a webcam connected to the computer running this app. "
               "Works when you run Streamlit locally in VS Code.")
    lc1, lc2 = st.columns([1, 3])
    cam_index = lc1.number_input("Camera number", min_value=0, max_value=10, value=0, step=1,
                                 help="0 is the default webcam. Try 1 or 2 for external cameras.")
    run_live = lc1.toggle("Start live camera", value=False)

    with lc2:
        frame_slot = st.empty()
        stats_slot = st.empty()

        if not run_live:
            frame_slot.info("Switch on 'Start live camera' to begin counting.")
        else:
            cap = cv2.VideoCapture(int(cam_index))
            if not cap.isOpened():
                st.error(f"Camera {int(cam_index)} could not be opened. Check that it is "
                         "connected and not used by another app.")
            else:
                try:
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            st.error("The camera stopped sending frames.")
                            break
                        t0 = time.perf_counter()
                        annotated, dets = run_detection(model, frame, conf, iou, imgsz, px_per_mm)
                        fps = 1 / max(time.perf_counter() - t0, 1e-6)
                        frame_slot.image(bgr_to_rgb(annotated), use_container_width=True)
                        stats_slot.markdown(
                            f"**Pipes in view:** {len(dets)} &nbsp;|&nbsp; **Speed:** {fps:.1f} frames/s"
                        )
                finally:
                    cap.release()

# --- Video file -------------------------------------------------------------
with tab_video:
    video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    step = st.slider("Analyse every Nth frame", 1, 30, 10,
                     help="Higher numbers process faster but check fewer frames.")
    if video is not None and st.button("Analyse video", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video.name).suffix) as tmp:
            tmp.write(video.getbuffer())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        progress = st.progress(0.0, text="Analysing video...")
        preview = st.empty()
        counts: list[int] = []
        best_frame, best_count = None, -1
        frame_no = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_no % step == 0:
                annotated, dets = run_detection(model, frame, conf, iou, imgsz, px_per_mm)
                counts.append(len(dets))
                if len(dets) > best_count:
                    best_count, best_frame = len(dets), annotated
                preview.image(bgr_to_rgb(annotated), use_container_width=True)
            frame_no += 1
            progress.progress(min(frame_no / total_frames, 1.0), text="Analysing video...")
        cap.release()
        progress.empty()

        if counts:
            v1, v2, v3 = st.columns(3)
            v1.markdown(stat_card("Frames analysed", len(counts)), unsafe_allow_html=True)
            v2.markdown(stat_card("Highest count in a frame", max(counts), accent=True), unsafe_allow_html=True)
            v3.markdown(stat_card("Average per frame", f"{sum(counts) / len(counts):.1f}"), unsafe_allow_html=True)
            st.line_chart(pd.DataFrame({"Pipes": counts}), height=220)
            if best_frame is not None:
                st.image(bgr_to_rgb(best_frame), caption="Frame with the most pipes detected",
                         use_container_width=True)
            st.session_state.history.append(
                {"id": f"video-{video.name}-{time.time()}", "Time": datetime.now().strftime("%H:%M:%S"),
                 "Source": video.name, "Pipes": max(counts), "Avg confidence": "-"}
            )
        else:
            st.error("No frames could be read from this video.")

# --- History ----------------------------------------------------------------
with tab_history:
    if not st.session_state.history:
        st.info("Your counts from this session appear here after you analyse an image or video.")
    else:
        hist = pd.DataFrame(st.session_state.history).drop(columns=["id"])
        st.dataframe(hist, use_container_width=True, hide_index=True)
        h1, h2 = st.columns(2)
        h1.download_button("Download history (CSV)", hist.to_csv(index=False).encode(),
                           file_name="pipe_count_history.csv", mime="text/csv",
                           use_container_width=True)
        if h2.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()
