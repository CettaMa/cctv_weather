#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI Maker Studio — Modern Telemetry Annotation Tool for CCTV Speed Measurement.
Revamped for Bandung CCTV & Weather Platform with CustomTkinter & High-Res Canvas.

Key Features:
- Modern Cyber/Telemetry Dark & Light visual design system.
- Instant 1-click video loading with smooth scroll retention (no list reload jumps).
- Exact 1:1 pixel-perfect interactive 4-point polygon annotation with live vertex dragging.
- Precision Magnifier / Loupe HUD for surgical lane marking alignment.
- Complete Video Player: Play/Pause loop, frame-by-frame stepping (-10, -1, +1, +10), scrub slider.
- Camera Perspective compass dial & quick cardinal direction presets (0° - 360°).
- Dimension presets (Lane: 3.5m, Zone: 15m, Car: 4.5m, Bus: 12m) & dynamic area telemetry.
- Multi-ROI management with live visual preview, selection, and deletion.
- Snapshot exporter: Save high-resolution annotated frames with overlays to outputs/snapshots/.
- Full backward compatibility with existing outputs/roi_annotations.json format.
"""

import json
import math
import os
import sys
import time
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

# ==========================================
# Paths & Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = os.path.join(BASE_DIR, "cctv_video_samples")
OUTPUT_JSON = os.path.join(BASE_DIR, "outputs", "roi_annotations.json")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "outputs", "snapshots")

MAX_POINTS = 4
MIN_POLY_AREA_PX = 400          # minimum polygon area in pixels^2
VERTEX_HIT_RADIUS_PX = 14       # display radius to detect vertex hover/drag

# ==========================================
# Telemetry Color Palette (RGB / Hex / BGR)
# ==========================================
THEME_DARK = {
    "bg_main": "#0B0F19",
    "panel_bg": "#111827",
    "card_bg": "#1E293B",
    "card_bg_hover": "#273549",
    "card_bg_active": "#1D4ED8",
    "card_border": "#334155",
    "card_border_focus": "#3B82F6",
    "text_primary": "#F8FAFC",
    "text_secondary": "#94A3B8",
    "text_muted": "#64748B",
    "accent_blue": "#3B82F6",
    "accent_blue_hover": "#2563EB",
    "accent_emerald": "#10B981",
    "accent_emerald_hover": "#059669",
    "accent_amber": "#F59E0B",
    "accent_amber_hover": "#D97706",
    "accent_crimson": "#EF4444",
    "accent_crimson_hover": "#DC2626",
    "accent_purple": "#8B5CF6",
    "canvas_bg": "#070A10",
    "hud_bg": "#111827",
}

THEME_LIGHT = {
    "bg_main": "#F8FAFC",
    "panel_bg": "#FFFFFF",
    "card_bg": "#F1F5F9",
    "card_bg_hover": "#E2E8F0",
    "card_bg_active": "#BFDBFE",
    "card_border": "#CBD5E1",
    "card_border_focus": "#3B82F6",
    "text_primary": "#0F172A",
    "text_secondary": "#475569",
    "text_muted": "#94A3B8",
    "accent_blue": "#2563EB",
    "accent_blue_hover": "#1D4ED8",
    "accent_emerald": "#059669",
    "accent_emerald_hover": "#047857",
    "accent_amber": "#D97706",
    "accent_amber_hover": "#B45309",
    "accent_crimson": "#DC2626",
    "accent_crimson_hover": "#B91C1C",
    "accent_purple": "#7C3AED",
    "canvas_bg": "#1E293B",
    "hud_bg": "#FFFFFF",
}

def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

def hex_to_bgr(h: str) -> tuple[int, int, int]:
    r, g, b = hex_to_rgb(h)
    return (b, g, r)

# OpenCV Overlay Colors (BGR)
CLR_WIDTH_BGR    = (11, 158, 245)    # Amber Gold #F59E0B (Width 1-2, 3-4)
CLR_LENGTH_BGR   = (246, 130, 59)    # Electric Blue #3B82F6 (Length 2-3, 4-1)
CLR_VERTEX_BGR   = (246, 130, 59)    # Blue vertex center
CLR_ACTIVE_BGR   = (16, 185, 129)    # Emerald active ring #10B981
CLR_PERSP_BGR    = (246, 92, 139)    # Purple perspective arrow #8B5CF6
CLR_FILL_BGR     = (59, 130, 246)    # Fill tint


# ==========================================
# Geometry & Validation Utilities
# ==========================================
def polygon_area_px(points: list[list[int] | tuple[int, int]]) -> float:
    """Calculate polygon area in pixels^2 via Shoelace formula."""
    if len(points) < 3:
        return 0.0
    pts = np.array(points, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def segments_intersect(p1, p2, p3, p4) -> bool:
    """Check if segment p1-p2 intersects with segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def sign(a, b, c):
        v = cross(a, b, c)
        return (v > 0) - (v < 0)

    d1 = sign(p3, p4, p1)
    d2 = sign(p3, p4, p2)
    d3 = sign(p1, p2, p3)
    d4 = sign(p1, p2, p4)
    return (d1 * d2 < 0) and (d3 * d4 < 0)

def polygon_is_valid(points: list[list[int] | tuple[int, int]]) -> tuple[bool, str]:
    """Validate 4-point polygon: non-intersecting (no bowtie) & sufficient area."""
    if len(points) != MAX_POINTS:
        return False, f"Memerlukan tepat {MAX_POINTS} titik poligon (saat ini {len(points)})."

    # 1. Check self-intersection (bowtie) first
    idx = range(MAX_POINTS)
    for i in idx:
        for j in idx:
            if j in ((i - 1) % MAX_POINTS, i, (i + 1) % MAX_POINTS):
                continue
            if segments_intersect(points[i], points[(i + 1) % MAX_POINTS],
                                  points[j], points[(j + 1) % MAX_POINTS]):
                return False, "Sisi poligon saling bersilangan (bowtie). Silakan geser titik."

    # 2. Check area threshold
    area = polygon_area_px(points)
    if area < MIN_POLY_AREA_PX:
        return False, f"Luas poligon terlalu kecil ({area:.0f} px² < {MIN_POLY_AREA_PX} px²)."

    return True, ""


# ==========================================
# Data Persistence Utilities
# ==========================================
def list_videos() -> list[str]:
    """List all CCTV video files in sample directory."""
    if not os.path.isdir(VIDEO_DIR):
        return []
    return sorted(
        f for f in os.listdir(VIDEO_DIR)
        if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov"))
    )

def load_existing() -> dict:
    """Load existing ROI annotations JSON."""
    if not os.path.exists(OUTPUT_JSON):
        return {}
    try:
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    if isinstance(data, list):
        data = {e["cctv_id"]: e for e in data if isinstance(e, dict) and "cctv_id" in e}
    return data

def save_json(data: dict):
    """Atomic write to JSON with .bak backup preservation."""
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    tmp_path = OUTPUT_JSON + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    if os.path.exists(OUTPUT_JSON):
        bak_path = OUTPUT_JSON + ".bak"
        try:
            os.replace(OUTPUT_JSON, bak_path)
        except OSError:
            pass
    os.replace(tmp_path, OUTPUT_JSON)


# ==========================================
# Main Application Class
# ==========================================
class ModernRoiMakerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Appearance & Window Configuration
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.theme_mode = "dark"
        self.theme = THEME_DARK

        self.title("PELINDUNG ROI STUDIO — Bandung CCTV Speed Telemetry")
        self.geometry("1420x880")
        self.minsize(1120, 700)
        self.configure(fg_color=self.theme["bg_main"])

        # Data Model
        self.store: dict = load_existing()
        self.video_name: str | None = None
        self.cctv_id: str | None = None
        self.entry: dict | None = None

        # Video & Frame State
        self.cap: cv2.VideoCapture | None = None
        self.frame: np.ndarray | None = None
        self.total_frames: int = 0
        self.current_frame_idx: int = 0
        self.video_fps: float = 30.0
        self.is_playing: bool = False
        self._play_after_id = None

        # Canvas & Interaction State
        self.points: list[list[int]] = []          # [[x, y], ...] in native frame pixels
        self.hover_vertex_idx: int | None = None   # Vertex index currently under mouse
        self.drag_vertex_idx: int | None = None    # Vertex index currently being dragged
        self.zoom: float = 1.0
        self.pan_x: float = 0.0                    # Frame coordinate pan offset
        self.pan_y: float = 0.0
        self._pan_start: tuple[int, int] | None = None
        self.mouse_frame_pos: tuple[int, int] | None = None
        self.show_loupe: bool = True
        self.active_roi_idx: int | None = None

        # Video List Item Widget Registry for O(1) highlight updates without list rebuilding
        self._video_card_widgets: dict[str, tuple[tk.Frame, tk.Frame, tk.Label, tk.Label]] = {}
        self.current_filter: str = "all"
        self._search_debounce_job = None

        # PhotoImage Reference to prevent GC
        self._photo_image = None

        # Build UI
        self._setup_layout()
        self._bind_global_shortcuts()

        # Initial video list population
        self.refresh_video_list()
        self.update_header_stats()

        # Auto-load first video if available
        vids = list_videos()
        if vids:
            self.load_video_by_name(vids[0])
        else:
            self.set_status("Selamat datang di ROI Studio. Tidak ada video di folder.", "warn")

        # Cleanup on close
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # -------------------------------------------------------------------------
    # Layout Builder
    # -------------------------------------------------------------------------
    def _setup_layout(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # 1. Header Bar (Row 0, Spanning all columns)
        self._build_header_bar()

        # 2. Main Content Split (Row 1)
        self._build_left_panel()
        self._build_center_panel()
        self._build_right_panel()

    # -------------------------------------------------------------------------
    # Header Bar
    # -------------------------------------------------------------------------
    def _build_header_bar(self):
        self.header_frame = ctk.CTkFrame(
            self, height=54, corner_radius=0,
            fg_color=self.theme["panel_bg"],
            border_width=1, border_color=self.theme["card_border"]
        )
        self.header_frame.grid(row=0, column=0, columnspan=3, sticky="new")
        self.header_frame.grid_propagate(False)
        self.header_frame.grid_columnconfigure(1, weight=1)

        # Left Branding
        brand_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        brand_frame.grid(row=0, column=0, padx=16, pady=8, sticky="w")

        title_lbl = ctk.CTkLabel(
            brand_frame, text="🛰️ PELINDUNG ROI STUDIO",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            text_color=self.theme["text_primary"]
        )
        title_lbl.pack(side="left")

        sub_badge = ctk.CTkLabel(
            brand_frame, text="BANDUNG CCTV TELEMETRY",
            font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
            text_color=self.theme["accent_blue"],
            fg_color="#1E293B" if self.theme_mode == "dark" else "#E2E8F0",
            corner_radius=4, padx=8, pady=2
        )
        sub_badge.pack(side="left", padx=10)

        # Center Stats Chips
        self.stats_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        self.stats_frame.grid(row=0, column=1, pady=8)

        self.stat_videos_chip = ctk.CTkLabel(
            self.stats_frame, text="📹 0 Videos",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=self.theme["text_secondary"],
            fg_color=self.theme["card_bg"],
            corner_radius=6, padx=10, pady=3
        )
        self.stat_videos_chip.pack(side="left", padx=4)

        self.stat_annotated_chip = ctk.CTkLabel(
            self.stats_frame, text="✓ 0 CCTV Dianotasi",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=self.theme["accent_emerald"],
            fg_color=self.theme["card_bg"],
            corner_radius=6, padx=10, pady=3
        )
        self.stat_annotated_chip.pack(side="left", padx=4)

        self.stat_rois_chip = ctk.CTkLabel(
            self.stats_frame, text="🏷️ 0 Total ROI",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=self.theme["accent_amber"],
            fg_color=self.theme["card_bg"],
            corner_radius=6, padx=10, pady=3
        )
        self.stat_rois_chip.pack(side="left", padx=4)

        # Right Header Actions
        actions_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        actions_frame.grid(row=0, column=2, padx=16, pady=8, sticky="e")

        snap_btn = ctk.CTkButton(
            actions_frame, text="📸 Snapshot", width=95, height=30,
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            fg_color=self.theme["card_bg"], hover_color=self.theme["card_bg_hover"],
            text_color=self.theme["text_primary"], border_width=1,
            border_color=self.theme["card_border"], corner_radius=6,
            command=lambda: self.export_snapshot(silent=False)
        )
        snap_btn.pack(side="left", padx=4)

        theme_btn = ctk.CTkButton(
            actions_frame, text="🌓 Tema", width=75, height=30,
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            fg_color=self.theme["card_bg"], hover_color=self.theme["card_bg_hover"],
            text_color=self.theme["text_primary"], border_width=1,
            border_color=self.theme["card_border"], corner_radius=6,
            command=self.toggle_theme
        )
        theme_btn.pack(side="left", padx=4)

        help_btn = ctk.CTkButton(
            actions_frame, text="❓ Shortcut", width=85, height=30,
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            fg_color=self.theme["card_bg"], hover_color=self.theme["card_bg_hover"],
            text_color=self.theme["text_primary"], border_width=1,
            border_color=self.theme["card_border"], corner_radius=6,
            command=self.show_shortcuts_modal
        )
        help_btn.pack(side="left", padx=4)

    # -------------------------------------------------------------------------
    # Left Panel: Video Explorer
    # -------------------------------------------------------------------------
    def _build_left_panel(self):
        self.left_panel = ctk.CTkFrame(
            self, width=310, corner_radius=0,
            fg_color=self.theme["panel_bg"],
            border_width=1, border_color=self.theme["card_border"]
        )
        self.left_panel.grid(row=1, column=0, sticky="nsew")
        self.left_panel.grid_propagate(False)
        self.left_panel.grid_rowconfigure(3, weight=1)
        self.left_panel.grid_columnconfigure(0, weight=1)

        # 1. Section Title
        sec_title = ctk.CTkLabel(
            self.left_panel, text="DAFTAR VIDEO CCTV",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=self.theme["accent_blue"]
        )
        sec_title.grid(row=0, column=0, padx=14, pady=(12, 6), sticky="w")

        # 2. Search Input
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", self._on_search_changed)

        search_wrap = ctk.CTkFrame(
            self.left_panel, fg_color=self.theme["card_bg"],
            border_width=1, border_color=self.theme["card_border"],
            corner_radius=8, height=36
        )
        search_wrap.grid(row=1, column=0, padx=14, pady=4, sticky="ew")
        search_wrap.pack_propagate(False)

        search_icon = ctk.CTkLabel(
            search_wrap, text="🔍", font=ctk.CTkFont(size=11),
            text_color=self.theme["text_muted"]
        )
        search_icon.pack(side="left", padx=(8, 4))

        self.search_entry = ctk.CTkEntry(
            search_wrap, textvariable=self.search_var,
            placeholder_text="Cari nama CCTV / jalan...",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            fg_color="transparent", border_width=0,
            text_color=self.theme["text_primary"],
            placeholder_text_color=self.theme["text_muted"]
        )
        self.search_entry.pack(side="left", fill="both", expand=True, padx=4)

        # 3. Filter Segmented Button
        self.filter_seg = ctk.CTkSegmentedButton(
            self.left_panel, values=["Semua", "Dianotasi", "Pending"],
            command=self._on_filter_changed,
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            selected_color=self.theme["accent_blue"],
            selected_hover_color=self.theme["accent_blue_hover"],
            unselected_color=self.theme["card_bg"],
            unselected_hover_color=self.theme["card_bg_hover"],
            corner_radius=6, height=28
        )
        self.filter_seg.set("Semua")
        self.filter_seg.grid(row=2, column=0, padx=14, pady=(8, 6), sticky="ew")

        # 4. Scrollable Container with native Tk scroll Canvas for ultra performance
        list_outer = tk.Frame(self.left_panel, bg=self.theme["panel_bg"])
        list_outer.grid(row=3, column=0, padx=8, pady=4, sticky="nsew")
        list_outer.grid_rowconfigure(0, weight=1)
        list_outer.grid_columnconfigure(0, weight=1)

        self.vlist_canvas = tk.Canvas(
            list_outer, bg=self.theme["panel_bg"],
            highlightthickness=0, bd=0
        )
        self.vlist_scrollbar = ctk.CTkScrollbar(
            list_outer, command=self.vlist_canvas.yview
        )
        self.video_items_container = tk.Frame(self.vlist_canvas, bg=self.theme["panel_bg"])

        self.video_items_container.bind(
            "<Configure>",
            lambda e: self.vlist_canvas.configure(scrollregion=self.vlist_canvas.bbox("all"))
        )
        self._vlist_win_id = self.vlist_canvas.create_window((0, 0), window=self.video_items_container, anchor="nw")
        self.vlist_canvas.configure(yscrollcommand=self.vlist_scrollbar.set)

        self.vlist_canvas.bind(
            "<Configure>",
            lambda e: self.vlist_canvas.itemconfig(self._vlist_win_id, width=e.width)
        )
        # Mousewheel scroll on list
        self.vlist_canvas.bind("<MouseWheel>", lambda e: self.vlist_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
        self.video_items_container.bind("<MouseWheel>", lambda e: self.vlist_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        self.vlist_canvas.grid(row=0, column=0, sticky="nsew")
        self.vlist_scrollbar.grid(row=0, column=1, sticky="ns")

        # 5. Load CCTV Button (optional quick CTA)
        bottom_left = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        bottom_left.grid(row=4, column=0, padx=14, pady=10, sticky="ew")

        self.load_btn = ctk.CTkButton(
            bottom_left, text="▶ Muat Video Terpilih",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            fg_color=self.theme["accent_blue"],
            hover_color=self.theme["accent_blue_hover"],
            text_color="#FFFFFF", corner_radius=8, height=36,
            command=self.load_selected_video
        )
        self.load_btn.pack(fill="x")

    # -------------------------------------------------------------------------
    # Center Panel: Media Viewport & Scrubber
    # -------------------------------------------------------------------------
    def _build_center_panel(self):
        self.center_panel = ctk.CTkFrame(
            self, corner_radius=0,
            fg_color=self.theme["bg_main"],
            border_width=0
        )
        self.center_panel.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        self.center_panel.grid_rowconfigure(1, weight=1)
        self.center_panel.grid_columnconfigure(0, weight=1)

        # 1. Viewport HUD Bar (Top of Center)
        self.viewport_hud = ctk.CTkFrame(
            self.center_panel, height=36, corner_radius=8,
            fg_color=self.theme["panel_bg"],
            border_width=1, border_color=self.theme["card_border"]
        )
        self.viewport_hud.grid(row=0, column=0, sticky="ew", padx=8, pady=(4, 4))
        self.viewport_hud.grid_propagate(False)

        # Video Title in HUD
        self.hud_video_title = ctk.CTkLabel(
            self.viewport_hud, text="Belum ada video dimuat",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=self.theme["text_primary"]
        )
        self.hud_video_title.pack(side="left", padx=12)

        # Resolution & FPS chip
        self.hud_res_chip = ctk.CTkLabel(
            self.viewport_hud, text="— × —",
            font=ctk.CTkFont(family="Consolas", size=10, weight="bold"),
            text_color=self.theme["text_secondary"],
            fg_color=self.theme["card_bg"], corner_radius=4, padx=6, pady=2
        )
        self.hud_res_chip.pack(side="left", padx=4)

        # Zoom Readout
        self.hud_zoom_chip = ctk.CTkLabel(
            self.viewport_hud, text="Zoom: 100%",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color=self.theme["text_secondary"],
            fg_color=self.theme["card_bg"], corner_radius=4, padx=6, pady=2
        )
        self.hud_zoom_chip.pack(side="left", padx=4)

        # Loupe Toggle Button
        self.loupe_toggle_btn = ctk.CTkButton(
            self.viewport_hud, text="🔍 Loupe: ON", width=84, height=24,
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            fg_color=self.theme["card_bg"], hover_color=self.theme["card_bg_hover"],
            text_color=self.theme["accent_emerald"], border_width=1,
            border_color=self.theme["card_border"], corner_radius=4,
            command=self.toggle_loupe
        )
        self.loupe_toggle_btn.pack(side="right", padx=(4, 8))

        # Mouse Coordinate HUD
        self.hud_coord_lbl = ctk.CTkLabel(
            self.viewport_hud, text="X: — | Y: —",
            font=ctk.CTkFont(family="Consolas", size=10, weight="bold"),
            text_color=self.theme["accent_blue"]
        )
        self.hud_coord_lbl.pack(side="right", padx=12)

        # 2. Main Interactive Canvas Container
        canvas_container = ctk.CTkFrame(
            self.center_panel, corner_radius=8,
            fg_color=self.theme["canvas_bg"],
            border_width=1, border_color=self.theme["card_border"]
        )
        canvas_container.grid(row=1, column=0, sticky="nsew", padx=8, pady=2)
        canvas_container.grid_rowconfigure(0, weight=1)
        canvas_container.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            canvas_container, bg=self.theme["canvas_bg"],
            highlightthickness=0, cursor="crosshair"
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Canvas Floating Zoom Controls (Bottom-Right overlay)
        zoom_overlay = ctk.CTkFrame(
            canvas_container, fg_color=self.theme["panel_bg"],
            border_width=1, border_color=self.theme["card_border"],
            corner_radius=6
        )
        zoom_overlay.place(relx=0.98, rely=0.96, anchor="se")

        btn_in = ctk.CTkButton(
            zoom_overlay, text="+", width=28, height=26,
            font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
            fg_color="transparent", hover_color=self.theme["card_bg_hover"],
            text_color=self.theme["text_primary"], corner_radius=4,
            command=lambda: self.zoom_at_center(1.25)
        )
        btn_in.pack(side="left", padx=1, pady=1)

        btn_out = ctk.CTkButton(
            zoom_overlay, text="−", width=28, height=26,
            font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
            fg_color="transparent", hover_color=self.theme["card_bg_hover"],
            text_color=self.theme["text_primary"], corner_radius=4,
            command=lambda: self.zoom_at_center(0.8)
        )
        btn_out.pack(side="left", padx=1, pady=1)

        btn_fit = ctk.CTkButton(
            zoom_overlay, text="Fit", width=36, height=26,
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            fg_color="transparent", hover_color=self.theme["card_bg_hover"],
            text_color=self.theme["text_primary"], corner_radius=4,
            command=self.reset_view
        )
        btn_fit.pack(side="left", padx=1, pady=1)

        # Canvas Bindings
        self.canvas.bind("<Button-1>", self.on_canvas_left_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_left_release)
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        self.canvas.bind("<Leave>", self.on_canvas_leave)
        self.canvas.bind("<Configure>", lambda _e: self.redraw())

        # Zoom & Pan Bindings
        self.canvas.bind("<MouseWheel>", self.on_wheel)          # Windows
        self.canvas.bind("<Button-4>", self.on_wheel)            # Linux
        self.canvas.bind("<Button-5>", self.on_wheel)
        for seq in ("<ButtonPress-3>", "<ButtonPress-2>"):
            self.canvas.bind(seq, self.on_pan_start)
        for seq in ("<B3-Motion>", "<B2-Motion>"):
            self.canvas.bind(seq, self.on_pan_move)
        for seq in ("<ButtonRelease-3>", "<ButtonRelease-2>"):
            self.canvas.bind(seq, self.on_pan_end)

        # 3. Media Player & Scrubber Bar
        self._build_scrubber_bar()

    def _build_scrubber_bar(self):
        scrub_frame = ctk.CTkFrame(
            self.center_panel, height=72, corner_radius=8,
            fg_color=self.theme["panel_bg"],
            border_width=1, border_color=self.theme["card_border"]
        )
        scrub_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=(4, 6))
        scrub_frame.grid_propagate(False)
        scrub_frame.grid_columnconfigure(1, weight=1)

        # Row 0: Slider & Timecode
        time_row = ctk.CTkFrame(scrub_frame, fg_color="transparent")
        time_row.pack(fill="x", padx=12, pady=(6, 2))

        self.timecode_lbl = ctk.CTkLabel(
            time_row, text="FRAME 000000 / 000000  •  00:00.0 / 00:00.0",
            font=ctk.CTkFont(family="Consolas", size=10, weight="bold"),
            text_color=self.theme["text_secondary"]
        )
        self.timecode_lbl.pack(side="left")

        # Row 1: Scrub Slider
        self.frame_slider = ctk.CTkSlider(
            scrub_frame, from_=0, to=1, number_of_steps=1,
            command=self.on_scrub_slider,
            button_color=self.theme["accent_blue"],
            button_hover_color=self.theme["accent_blue_hover"],
            progress_color=self.theme["accent_blue"],
            fg_color=self.theme["card_bg"],
            height=14
        )
        self.frame_slider.set(0)
        self.frame_slider.pack(fill="x", padx=12, pady=(0, 4))

        # Row 2: Control Buttons
        btn_row = ctk.CTkFrame(scrub_frame, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=(0, 6))

        # Step First
        ctk.CTkButton(
            btn_row, text="⏮", width=32, height=26,
            font=ctk.CTkFont(size=11), fg_color=self.theme["card_bg"],
            hover_color=self.theme["card_bg_hover"],
            text_color=self.theme["text_primary"], corner_radius=5,
            command=lambda: self.seek_frame(0)
        ).pack(side="left", padx=2)

        # Step -10
        ctk.CTkButton(
            btn_row, text="◀ -10", width=50, height=26,
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            fg_color=self.theme["card_bg"], hover_color=self.theme["card_bg_hover"],
            text_color=self.theme["text_primary"], corner_radius=5,
            command=lambda: self.step_frame(-10)
        ).pack(side="left", padx=2)

        # Step -1
        ctk.CTkButton(
            btn_row, text="◂ -1", width=42, height=26,
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            fg_color=self.theme["card_bg"], hover_color=self.theme["card_bg_hover"],
            text_color=self.theme["text_primary"], corner_radius=5,
            command=lambda: self.step_frame(-1)
        ).pack(side="left", padx=2)

        # Play / Pause Main Button
        self.play_btn = ctk.CTkButton(
            btn_row, text="▶ Putar Video", width=110, height=26,
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            fg_color=self.theme["accent_emerald"],
            hover_color=self.theme["accent_emerald_hover"],
            text_color="#FFFFFF", corner_radius=5,
            command=self.toggle_play
        )
        self.play_btn.pack(side="left", padx=6)

        # Step +1
        ctk.CTkButton(
            btn_row, text="+1 ▸", width=42, height=26,
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            fg_color=self.theme["card_bg"], hover_color=self.theme["card_bg_hover"],
            text_color=self.theme["text_primary"], corner_radius=5,
            command=lambda: self.step_frame(1)
        ).pack(side="left", padx=2)

        # Step +10
        ctk.CTkButton(
            btn_row, text="+10 ▶", width=50, height=26,
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            fg_color=self.theme["card_bg"], hover_color=self.theme["card_bg_hover"],
            text_color=self.theme["text_primary"], corner_radius=5,
            command=lambda: self.step_frame(10)
        ).pack(side="left", padx=2)

        # Step Last
        ctk.CTkButton(
            btn_row, text="⏭", width=32, height=26,
            font=ctk.CTkFont(size=11), fg_color=self.theme["card_bg"],
            hover_color=self.theme["card_bg_hover"],
            text_color=self.theme["text_primary"], corner_radius=5,
            command=lambda: self.seek_frame(max(0, self.total_frames - 1))
        ).pack(side="left", padx=2)

        # Playback Speed selector
        self.speed_menu = ctk.CTkOptionMenu(
            btn_row, values=["0.5x", "1.0x", "2.0x"], width=72, height=26,
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            fg_color=self.theme["card_bg"],
            button_color=self.theme["card_bg_hover"],
            button_hover_color=self.theme["accent_blue"],
            text_color=self.theme["text_primary"], corner_radius=5
        )
        self.speed_menu.set("1.0x")
        self.speed_menu.pack(side="right", padx=2)

        speed_lbl = ctk.CTkLabel(
            btn_row, text="Kecepatan:",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color=self.theme["text_muted"]
        )
        speed_lbl.pack(side="right", padx=4)

    # -------------------------------------------------------------------------
    # Right Panel: Telemetry & ROI Inspector
    # -------------------------------------------------------------------------
    def _build_right_panel(self):
        self.right_panel = ctk.CTkFrame(
            self, width=340, corner_radius=0,
            fg_color=self.theme["panel_bg"],
            border_width=1, border_color=self.theme["card_border"]
        )
        self.right_panel.grid(row=1, column=2, sticky="nsew")
        self.right_panel.grid_propagate(False)
        self.right_panel.grid_columnconfigure(0, weight=1)

        # Scrollable container for inspector controls
        self.inspector_scroll = ctk.CTkScrollableFrame(
            self.right_panel, fg_color="transparent", corner_radius=0
        )
        self.inspector_scroll.pack(fill="both", expand=True, padx=8, pady=8)

        # Card 1: Camera Perspective
        self._build_perspective_card()

        # Card 2: Active ROI Configuration & Actions
        self._build_roi_editor_card()

        # Card 3: Saved ROIs in CCTV
        self._build_saved_rois_card()

        # Status & Guidance Banner (Bottom of Right Panel)
        self._build_status_banner()

    def _build_perspective_card(self):
        persp_card = ctk.CTkFrame(
            self.inspector_scroll, fg_color=self.theme["card_bg"],
            border_width=1, border_color=self.theme["card_border"],
            corner_radius=8
        )
        persp_card.pack(fill="x", pady=(0, 8))

        # Title
        hdr = ctk.CTkFrame(persp_card, fg_color="transparent")
        hdr.pack(fill="x", padx=12, pady=(10, 4))
        ctk.CTkLabel(
            hdr, text="🧭 PERSPEKTIF KAMERA",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=self.theme["accent_purple"]
        ).pack(side="left")

        self.persp_deg_val_lbl = ctk.CTkLabel(
            hdr, text="Belum diset",
            font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
            text_color=self.theme["text_secondary"]
        )
        self.persp_deg_val_lbl.pack(side="right")

        # Angle Slider
        self.persp_slider = ctk.CTkSlider(
            persp_card, from_=0, to=360, number_of_steps=360,
            command=self.on_persp_slider_change,
            button_color=self.theme["accent_purple"],
            progress_color=self.theme["accent_purple"],
            fg_color=self.theme["panel_bg"], height=14
        )
        self.persp_slider.set(0)
        self.persp_slider.pack(fill="x", padx=12, pady=6)

        # Quick Cardinal Presets
        presets_frame = ctk.CTkFrame(persp_card, fg_color="transparent")
        presets_frame.pack(fill="x", padx=12, pady=(0, 10))
        presets_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        cardinals = [("T (0°)", 0), ("U (90°)", 90), ("B (180°)", 180), ("S (270°)", 270)]
        for col, (label, deg) in enumerate(cardinals):
            btn = ctk.CTkButton(
                presets_frame, text=label, height=24,
                font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
                fg_color=self.theme["panel_bg"],
                hover_color=self.theme["card_bg_hover"],
                text_color=self.theme["text_primary"],
                border_width=1, border_color=self.theme["card_border"],
                corner_radius=4,
                command=lambda d=deg: self.set_perspective(d)
            )
            btn.grid(row=0, column=col, padx=2, sticky="ew")

    def _build_roi_editor_card(self):
        roi_card = ctk.CTkFrame(
            self.inspector_scroll, fg_color=self.theme["card_bg"],
            border_width=1, border_color=self.theme["card_border"],
            corner_radius=8
        )
        roi_card.pack(fill="x", pady=(0, 8))

        # Header
        hdr = ctk.CTkFrame(roi_card, fg_color="transparent")
        hdr.pack(fill="x", padx=12, pady=(10, 4))

        ctk.CTkLabel(
            hdr, text="📐 ANOTASI ROI AKTIF",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=self.theme["accent_blue"]
        ).pack(side="left")

        self.roi_pts_counter_lbl = ctk.CTkLabel(
            hdr, text="0/4 Titik",
            font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
            text_color=self.theme["accent_amber"],
            fg_color=self.theme["panel_bg"], corner_radius=4, padx=6, pady=2
        )
        self.roi_pts_counter_lbl.pack(side="right")

        # Name Field
        name_wrap = ctk.CTkFrame(roi_card, fg_color="transparent")
        name_wrap.pack(fill="x", padx=12, pady=(6, 4))
        ctk.CTkLabel(
            name_wrap, text="Nama ROI:",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color=self.theme["text_secondary"]
        ).pack(anchor="w")

        self.roi_name_var = tk.StringVar(value="roi_1")
        self.roi_name_entry = ctk.CTkEntry(
            name_wrap, textvariable=self.roi_name_var, height=32,
            font=ctk.CTkFont(family="Segoe UI", size=11),
            fg_color=self.theme["panel_bg"],
            border_width=1, border_color=self.theme["card_border"],
            text_color=self.theme["text_primary"]
        )
        self.roi_name_entry.pack(fill="x", pady=(2, 0))

        # Dimensions Grid: Length & Width
        dim_grid = ctk.CTkFrame(roi_card, fg_color="transparent")
        dim_grid.pack(fill="x", padx=12, pady=4)
        dim_grid.grid_columnconfigure((0, 1), weight=1)

        # Length (Panjang)
        len_col = ctk.CTkFrame(dim_grid, fg_color="transparent")
        len_col.grid(row=0, column=0, padx=(0, 4), sticky="ew")

        ctk.CTkLabel(
            len_col, text="Panjang (m):",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color=self.theme["accent_blue"]
        ).pack(anchor="w")

        self.roi_len_var = tk.StringVar(value="")
        self.roi_len_entry = ctk.CTkEntry(
            len_col, textvariable=self.roi_len_var, height=32,
            font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
            fg_color=self.theme["panel_bg"],
            border_width=1, border_color=self.theme["card_border"],
            text_color=self.theme["text_primary"], placeholder_text="mis. 15.0"
        )
        self.roi_len_entry.pack(fill="x", pady=(2, 0))

        # Width (Lebar)
        wid_col = ctk.CTkFrame(dim_grid, fg_color="transparent")
        wid_col.grid(row=0, column=1, padx=(4, 0), sticky="ew")

        ctk.CTkLabel(
            wid_col, text="Lebar (m):",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color=self.theme["accent_amber"]
        ).pack(anchor="w")

        self.roi_wid_var = tk.StringVar(value="")
        self.roi_wid_entry = ctk.CTkEntry(
            wid_col, textvariable=self.roi_wid_var, height=32,
            font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
            fg_color=self.theme["panel_bg"],
            border_width=1, border_color=self.theme["card_border"],
            text_color=self.theme["text_primary"], placeholder_text="mis. 3.5"
        )
        self.roi_wid_entry.pack(fill="x", pady=(2, 0))

        # Quick Preset Chips
        chip_frame = ctk.CTkFrame(roi_card, fg_color="transparent")
        chip_frame.pack(fill="x", padx=12, pady=(4, 6))
        chip_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        chips = [("Lajur 3.5m", 15.0, 3.5), ("Simpang 20m", 20.0, 7.0),
                 ("Mobil 4.5m", 4.5, 2.0), ("Bus 12m", 12.0, 3.0)]
        for col, (label, l, w) in enumerate(chips):
            btn = ctk.CTkButton(
                chip_frame, text=label, height=22,
                font=ctk.CTkFont(family="Segoe UI", size=9),
                fg_color=self.theme["panel_bg"],
                hover_color=self.theme["card_bg_hover"],
                text_color=self.theme["text_secondary"],
                border_width=1, border_color=self.theme["card_border"],
                corner_radius=4,
                command=lambda length=l, width=w: self.apply_dim_preset(length, width)
            )
            btn.grid(row=0, column=col, padx=1, sticky="ew")

        # Telemetry Area & Aspect Ratio Readout
        self.telemetry_readout_lbl = ctk.CTkLabel(
            roi_card, text="Luas Tanah: — m²  |  Rasio: —",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color=self.theme["text_muted"]
        )
        self.telemetry_readout_lbl.pack(padx=12, pady=(2, 8))

        # Main Action Button: Save ROI
        self.save_roi_btn = ctk.CTkButton(
            roi_card, text="💾 SIMPAN ROI", height=38,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            fg_color=self.theme["accent_blue"],
            hover_color=self.theme["accent_blue_hover"],
            text_color="#FFFFFF", corner_radius=6,
            command=self.save_roi
        )
        self.save_roi_btn.pack(fill="x", padx=12, pady=(0, 8))

        # Auxiliary Buttons: New, Undo, Reset
        aux_frame = ctk.CTkFrame(roi_card, fg_color="transparent")
        aux_frame.pack(fill="x", padx=12, pady=(0, 10))
        aux_frame.grid_columnconfigure((0, 1, 2), weight=1)

        ctk.CTkButton(
            aux_frame, text="+ Baru", height=28,
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            fg_color=self.theme["panel_bg"], hover_color=self.theme["card_bg_hover"],
            text_color=self.theme["text_primary"], border_width=1,
            border_color=self.theme["card_border"], corner_radius=4,
            command=self.new_roi
        ).grid(row=0, column=0, padx=2, sticky="ew")

        ctk.CTkButton(
            aux_frame, text="↩ Undo", height=28,
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            fg_color=self.theme["panel_bg"], hover_color=self.theme["card_bg_hover"],
            text_color=self.theme["text_primary"], border_width=1,
            border_color=self.theme["card_border"], corner_radius=4,
            command=self.undo_point
        ).grid(row=0, column=1, padx=2, sticky="ew")

        ctk.CTkButton(
            aux_frame, text="✕ Reset", height=28,
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            fg_color=self.theme["panel_bg"], hover_color=self.theme["accent_crimson"],
            text_color=self.theme["accent_crimson"], border_width=1,
            border_color=self.theme["card_border"], corner_radius=4,
            command=self.reset_points
        ).grid(row=0, column=2, padx=2, sticky="ew")

    def _build_saved_rois_card(self):
        saved_card = ctk.CTkFrame(
            self.inspector_scroll, fg_color=self.theme["card_bg"],
            border_width=1, border_color=self.theme["card_border"],
            corner_radius=8
        )
        saved_card.pack(fill="x", pady=(0, 8))

        # Header
        hdr = ctk.CTkFrame(saved_card, fg_color="transparent")
        hdr.pack(fill="x", padx=12, pady=(10, 4))

        ctk.CTkLabel(
            hdr, text="📋 ROI TERSIMPAN DI CCTV INI",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=self.theme["accent_emerald"]
        ).pack(side="left")

        self.saved_rois_count_lbl = ctk.CTkLabel(
            hdr, text="0 Item",
            font=ctk.CTkFont(family="Consolas", size=10, weight="bold"),
            text_color=self.theme["text_muted"]
        )
        self.saved_rois_count_lbl.pack(side="right")

        # Container for saved ROI items
        self.saved_rois_list_frame = tk.Frame(saved_card, bg=self.theme["card_bg"])
        self.saved_rois_list_frame.pack(fill="x", padx=8, pady=(4, 10))

    def _build_status_banner(self):
        self.status_banner = ctk.CTkFrame(
            self.right_panel, height=64, corner_radius=8,
            fg_color=self.theme["card_bg"],
            border_width=1, border_color=self.theme["card_border"]
        )
        self.status_banner.pack(fill="x", side="bottom", padx=8, pady=8)
        self.status_banner.pack_propagate(False)

        self.status_icon_lbl = ctk.CTkLabel(
            self.status_banner, text="ℹ️", font=ctk.CTkFont(size=14)
        )
        self.status_icon_lbl.pack(side="left", padx=(10, 6))

        self.status_text_lbl = ctk.CTkLabel(
            self.status_banner, text="",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color=self.theme["text_secondary"],
            wraplength=260, justify="left", anchor="w"
        )
        self.status_text_lbl.pack(side="left", fill="both", expand=True, padx=(0, 8))

    # -------------------------------------------------------------------------
    # Global Key Bindings & Shortcuts
    # -------------------------------------------------------------------------
    def _bind_global_shortcuts(self):
        self.bind_all("<space>", lambda _e: self.toggle_play())
        self.bind_all("<Left>", lambda _e: self.step_frame(-1))
        self.bind_all("<Right>", lambda _e: self.step_frame(1))
        self.bind_all("<Shift-Left>", lambda _e: self.step_frame(-10))
        self.bind_all("<Shift-Right>", lambda _e: self.step_frame(10))
        self.bind_all("<Control-z>", lambda _e: self.undo_point())
        self.bind_all("<Control-s>", lambda _e: self.save_roi())
        self.bind_all("<Control-n>", lambda _e: self.new_roi())
        self.bind_all("<Escape>", lambda _e: self.reset_points())
        self.bind_all("<Control-equal>", lambda _e: self.zoom_at_center(1.25))
        self.bind_all("<Control-plus>", lambda _e: self.zoom_at_center(1.25))
        self.bind_all("<Control-minus>", lambda _e: self.zoom_at_center(0.8))
        self.bind_all("<Control-0>", lambda _e: self.reset_view())

    # -------------------------------------------------------------------------
    # High-Performance Video Explorer Logic
    # -------------------------------------------------------------------------
    def refresh_video_list(self):
        q = self.search_var.get().strip().lower()
        all_vids = list_videos()

        self._video_card_widgets.clear()
        for child in self.video_items_container.winfo_children():
            child.destroy()

        filtered_vids = []
        for v in all_vids:
            cid = os.path.splitext(v)[0]
            roi_count = len(self.store.get(cid, {}).get("rois", []))
            has_roi = roi_count > 0

            if self.current_filter == "annotated" and not has_roi:
                continue
            if self.current_filter == "pending" and has_roi:
                continue
            if q and q not in v.lower():
                continue

            filtered_vids.append((v, cid, roi_count))

        if not filtered_vids:
            empty_lbl = tk.Label(
                self.video_items_container, text="Tidak ada video yang cocok",
                font=("Segoe UI", 10), bg=self.theme["panel_bg"],
                fg=self.theme["text_muted"]
            )
            empty_lbl.pack(pady=20)
            return

        for v, cid, roi_count in filtered_vids:
            is_active = (self.video_name == v)
            card_bg = self.theme["card_bg_active"] if is_active else self.theme["card_bg"]
            text_fg = "#FFFFFF" if is_active else self.theme["text_primary"]

            card = tk.Frame(
                self.video_items_container,
                bg=card_bg,
                highlightthickness=1,
                highlightbackground=self.theme["accent_blue"] if is_active else self.theme["card_border"],
                cursor="hand2"
            )
            card.pack(fill="x", pady=2, padx=2)

            inner = tk.Frame(card, bg=card_bg)
            inner.pack(fill="both", expand=True, padx=8, pady=6)

            # Icon
            icon = "🎬" if roi_count > 0 else "📹"
            ic_lbl = tk.Label(inner, text=icon, font=("Segoe UI", 10), bg=card_bg, fg=text_fg)
            ic_lbl.pack(side="left", padx=(0, 6))

            # Video Name
            clean_name = v
            if len(clean_name) > 28:
                clean_name = clean_name[:25] + "..."
            name_lbl = tk.Label(
                inner, text=clean_name,
                font=("Segoe UI", 9, "bold" if is_active else "normal"),
                bg=card_bg, fg=text_fg, anchor="w"
            )
            name_lbl.pack(side="left", fill="x", expand=True)

            # Status Badge
            if roi_count > 0:
                badge_bg = "#064E3B" if self.theme_mode == "dark" else "#D1FAE5"
                badge = tk.Label(
                    inner, text=f"✓ {roi_count}",
                    font=("Segoe UI", 8, "bold"),
                    bg=badge_bg, fg=self.theme["accent_emerald"],
                    padx=4, pady=1
                )
            else:
                badge = tk.Label(
                    inner, text="○ 0",
                    font=("Segoe UI", 8),
                    bg=self.theme["panel_bg"], fg=self.theme["text_muted"],
                    padx=4, pady=1
                )
            badge.pack(side="right")

            self._video_card_widgets[v] = (card, inner, name_lbl, badge)

            # Single Click immediately loads that video!
            def on_click(_e, vid=v):
                self.load_video_by_name(vid)

            def on_enter(_e, vid=v):
                if self.video_name != vid and vid in self._video_card_widgets:
                    c, inn, _, _ = self._video_card_widgets[vid]
                    c.configure(bg=self.theme["card_bg_hover"])
                    inn.configure(bg=self.theme["card_bg_hover"])

            def on_leave(_e, vid=v):
                if self.video_name != vid and vid in self._video_card_widgets:
                    c, inn, _, _ = self._video_card_widgets[vid]
                    c.configure(bg=self.theme["card_bg"])
                    inn.configure(bg=self.theme["card_bg"])

            for widget in (card, inner, ic_lbl, name_lbl, badge):
                widget.bind("<Button-1>", on_click)
                widget.bind("<Enter>", on_enter)
                widget.bind("<Leave>", on_leave)

    def _update_list_selection_visuals(self, active_vid: str):
        """Update list selection highlight in O(1) without recreating widgets or scrolling."""
        for vid, (card, inner, name_lbl, badge) in self._video_card_widgets.items():
            is_active = (vid == active_vid)
            card_bg = self.theme["card_bg_active"] if is_active else self.theme["card_bg"]
            text_fg = "#FFFFFF" if is_active else self.theme["text_primary"]
            border_color = self.theme["accent_blue"] if is_active else self.theme["card_border"]

            card.configure(bg=card_bg, highlightbackground=border_color)
            inner.configure(bg=card_bg)
            name_lbl.configure(bg=card_bg, fg=text_fg, font=("Segoe UI", 9, "bold" if is_active else "normal"))

    def _on_search_changed(self, *_args):
        if self._search_debounce_job is not None:
            self.after_cancel(self._search_debounce_job)
        self._search_debounce_job = self.after(150, self.refresh_video_list)

    def _on_filter_changed(self, value):
        mapping = {"Semua": "all", "Dianotasi": "annotated", "Pending": "pending"}
        self.current_filter = mapping.get(value, "all")
        self.refresh_video_list()

    def select_video_by_name(self, vid_name: str):
        self.load_video_by_name(vid_name)

    def load_selected_video(self):
        target = getattr(self, "selected_target_video", None)
        if not target:
            vids = list_videos()
            if vids:
                target = vids[0]
        if target:
            self.load_video_by_name(target)
        else:
            self.set_status("Pilih video dari daftar terlebih dahulu.", "warn")

    def load_video_by_name(self, name: str):
        self.stop_play()
        path = os.path.join(VIDEO_DIR, name)
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        if not ok or frame is None:
            messagebox.showerror("Gagal Membaca Video", f"Tidak dapat membuka berkas video:\n{name}")
            return

        self.release_capture()
        self.cap = cap
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self.video_fps <= 0 or math.isnan(self.video_fps):
            self.video_fps = 30.0

        self.current_frame_idx = 0
        self.video_name = name
        self.selected_target_video = name
        self.cctv_id = os.path.splitext(name)[0]
        self.frame = frame

        h, w = frame.shape[:2]
        self.entry = self.store.get(self.cctv_id) or {
            "cctv_id": self.cctv_id,
            "video_name": name,
            "rois": [],
            "perspective_deg": None,
            "frame_width": int(w),
            "frame_height": int(h),
        }
        self.entry.setdefault("perspective_deg", None)
        self.entry.setdefault("frame_width", int(w))
        self.entry.setdefault("frame_height", int(h))
        self.store[self.cctv_id] = self.entry

        # Reset zoom and pan
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        # Update Scrubber & HUD
        max_idx = max(1, self.total_frames - 1)
        self.frame_slider.configure(to=max_idx, number_of_steps=max_idx)
        self.frame_slider.set(0)
        self.update_timecode_display()

        self.hud_video_title.configure(text=f"🎬 {name}")
        self.hud_res_chip.configure(text=f"{w}×{h} @ {self.video_fps:.0f}fps")

        # Load perspective
        persp = self.entry.get("perspective_deg")
        if persp is not None:
            self.persp_slider.set(persp)
            self.persp_deg_val_lbl.configure(text=f"{persp:g}°", text_color=self.theme["accent_purple"])
        else:
            self.persp_slider.set(0)
            self.persp_deg_val_lbl.configure(text="Belum diset", text_color=self.theme["text_secondary"])

        # Reset points for fresh annotation on loaded video
        self.new_roi()
        self.refresh_saved_rois_list()
        self._update_list_selection_visuals(name)
        self.update_header_stats()

        n_rois = len(self.entry.get("rois", []))
        self.set_status(f"Video dimuat: {name} ({w}×{h}) • {n_rois} ROI tersimpan. Klik 4 titik pada gambar.", "ok")
        self.redraw()

    def release_capture(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def on_close(self):
        self.stop_play()
        self.release_capture()
        self.destroy()

    # -------------------------------------------------------------------------
    # Video Playback & Scrubbing
    # -------------------------------------------------------------------------
    def toggle_play(self):
        if self.frame is None or self.cap is None:
            return
        if self.is_playing:
            self.stop_play()
        else:
            self.start_play()

    def start_play(self):
        if self.cap is None:
            return
        self.is_playing = True
        self.play_btn.configure(
            text="⏸ Jeda Video",
            fg_color=self.theme["accent_amber"],
            hover_color=self.theme["accent_amber_hover"]
        )
        self._play_loop()

    def stop_play(self):
        self.is_playing = False
        if self._play_after_id is not None:
            self.after_cancel(self._play_after_id)
            self._play_after_id = None
        self.play_btn.configure(
            text="▶ Putar Video",
            fg_color=self.theme["accent_emerald"],
            hover_color=self.theme["accent_emerald_hover"]
        )

    def _play_loop(self):
        if not self.is_playing or self.cap is None:
            return
        next_idx = self.current_frame_idx + 1
        if next_idx >= self.total_frames:
            next_idx = 0

        self.seek_frame(next_idx, redraw_now=True)

        speed_str = self.speed_menu.get()
        speed_mult = 1.0
        if "0.5" in speed_str:
            speed_mult = 0.5
        elif "2.0" in speed_str:
            speed_mult = 2.0

        delay_ms = max(10, int((1000.0 / self.video_fps) / speed_mult))
        self._play_after_id = self.after(delay_ms, self._play_loop)

    def on_scrub_slider(self, val):
        idx = int(round(val))
        self.seek_frame(idx)

    def step_frame(self, delta: int):
        if self.frame is None:
            return
        new_idx = max(0, min(self.current_frame_idx + delta, self.total_frames - 1))
        self.seek_frame(new_idx)

    def seek_frame(self, frame_idx: int, redraw_now: bool = True):
        if self.cap is None:
            return
        frame_idx = max(0, min(frame_idx, max(0, self.total_frames - 1)))
        self.current_frame_idx = frame_idx
        self.frame_slider.set(frame_idx)
        self.update_timecode_display()

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if ok and frame is not None:
            self.frame = frame
            if redraw_now:
                self.redraw()

    def update_timecode_display(self):
        cur_sec = self.current_frame_idx / max(1.0, self.video_fps)
        tot_sec = max(0, self.total_frames - 1) / max(1.0, self.video_fps)
        cur_m, cur_s = int(cur_sec // 60), cur_sec % 60
        tot_m, tot_s = int(tot_sec // 60), tot_sec % 60
        text = (f"FRAME {self.current_frame_idx:06d} / {max(0, self.total_frames - 1):06d}  •  "
                f"{cur_m:02d}:{cur_s:04.1f} / {tot_m:02d}:{tot_s:04.1f}")
        self.timecode_lbl.configure(text=text)

    # -------------------------------------------------------------------------
    # Canvas Viewport & Coordinate Conversions
    # -------------------------------------------------------------------------
    def _view_params(self) -> tuple[float, float, float]:
        """Compute base scale, zoom, and viewport offsets in canvas pixels."""
        if self.frame is None:
            return 1.0, 0.0, 0.0
        h, w = self.frame.shape[:2]
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        base = min(cw / w, ch / h, 1.0)
        scale = base * self.zoom
        disp_w, disp_h = w * scale, h * scale
        ox = (cw - disp_w) / 2.0 - self.pan_x * scale
        oy = (ch - disp_h) / 2.0 - self.pan_y * scale
        return scale, ox, oy

    def _canvas_to_frame(self, cx: int, cy: int) -> tuple[int, int] | None:
        """Convert canvas pixel coordinates to native video frame coordinates."""
        if self.frame is None:
            return None
        scale, ox, oy = self._view_params()
        fx = int(round((cx - ox) / scale))
        fy = int(round((cy - oy) / scale))
        h, w = self.frame.shape[:2]
        if not (0 <= fx < w and 0 <= fy < h):
            return None
        return fx, fy

    def _frame_to_canvas(self, fx: int, fy: int) -> tuple[int, int]:
        """Convert native video frame coordinates to canvas pixel coordinates."""
        scale, ox, oy = self._view_params()
        cx = int(round(fx * scale + ox))
        cy = int(round(fy * scale + oy))
        return cx, cy

    def _find_vertex_near_canvas(self, cx: int, cy: int) -> int | None:
        for i, pt in enumerate(self.points):
            px, py = self._frame_to_canvas(pt[0], pt[1])
            dist = math.hypot(cx - px, cy - py)
            if dist <= VERTEX_HIT_RADIUS_PX:
                return i
        return None

    # -------------------------------------------------------------------------
    # Canvas Event Handlers (Click, Drag, Motion, Zoom, Pan)
    # -------------------------------------------------------------------------
    def on_canvas_left_click(self, event):
        if self.frame is None:
            return

        # Check if clicking on an existing vertex to drag
        hit = self._find_vertex_near_canvas(event.x, event.y)
        if hit is not None:
            self.drag_vertex_idx = hit
            self.hover_vertex_idx = hit
            self.canvas.configure(cursor="fleur")
            self.redraw()
            return

        # Otherwise place new point if < MAX_POINTS
        fpos = self._canvas_to_frame(event.x, event.y)
        if fpos is None:
            return

        if len(self.points) < MAX_POINTS:
            self.points.append([fpos[0], fpos[1]])
            self.update_roi_point_counters()
            self.redraw()

    def on_canvas_left_drag(self, event):
        if self.frame is None or self.drag_vertex_idx is None:
            return
        fpos = self._canvas_to_frame(event.x, event.y)
        if fpos is not None:
            self.points[self.drag_vertex_idx] = [fpos[0], fpos[1]]
            self.mouse_frame_pos = fpos
            self.update_telemetry_metrics()
            self.redraw()

    def on_canvas_left_release(self, _event):
        if self.drag_vertex_idx is not None:
            self.drag_vertex_idx = None
            self.canvas.configure(cursor="crosshair")
            self.update_telemetry_metrics()
            self.redraw()

    def on_canvas_motion(self, event):
        if self.frame is None:
            return
        fpos = self._canvas_to_frame(event.x, event.y)
        self.mouse_frame_pos = fpos

        if fpos is not None:
            self.hud_coord_lbl.configure(text=f"X: {fpos[0]:04d} | Y: {fpos[1]:04d}")
        else:
            self.hud_coord_lbl.configure(text="X: — | Y: —")

        prev_hover = self.hover_vertex_idx
        hit = self._find_vertex_near_canvas(event.x, event.y)
        self.hover_vertex_idx = hit

        if hit is not None and self.drag_vertex_idx is None:
            self.canvas.configure(cursor="hand2")
        elif self.drag_vertex_idx is None:
            self.canvas.configure(cursor="crosshair")

        if prev_hover != self.hover_vertex_idx or (self.show_loupe and fpos is not None):
            self.redraw()

    def on_canvas_leave(self, _event):
        self.mouse_frame_pos = None
        self.hover_vertex_idx = None
        self.hud_coord_lbl.configure(text="X: — | Y: —")
        self.redraw()

    def on_wheel(self, event):
        if self.frame is None:
            return
        delta = getattr(event, "delta", 0)
        if event.num == 4:
            factor = 1.25
        elif event.num == 5:
            factor = 0.8
        else:
            factor = 1.25 if delta > 0 else 0.8
        self.zoom_at(factor, event.x, event.y)

    def zoom_at_center(self, factor: float):
        cw = max(self.canvas.winfo_width(), 100) // 2
        ch = max(self.canvas.winfo_height(), 100) // 2
        self.zoom_at(factor, cw, ch)

    def zoom_at(self, factor: float, cx: int, cy: int):
        if self.frame is None:
            return
        new_zoom = max(0.5, min(16.0, self.zoom * factor))
        if new_zoom == self.zoom:
            return

        old_scale, old_ox, old_oy = self._view_params()
        fx = (cx - old_ox) / old_scale
        fy = (cy - old_oy) / old_scale

        self.zoom = new_zoom
        new_scale = (min(self.canvas.winfo_width() / self.frame.shape[1],
                         self.canvas.winfo_height() / self.frame.shape[0], 1.0) * self.zoom)

        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        h, w = self.frame.shape[:2]

        center_off_x = (cw - w * new_scale) / 2.0
        center_off_y = (ch - h * new_scale) / 2.0
        self.pan_x = (fx * new_scale + center_off_x - cx) / new_scale
        self.pan_y = (fy * new_scale + center_off_y - cy) / new_scale

        self._clamp_pan()
        self.hud_zoom_chip.configure(text=f"Zoom: {int(self.zoom * 100)}%")
        self.redraw()

    def _clamp_pan(self):
        max_pan = 0.5 * (1.0 - 1.0 / self.zoom) if self.zoom > 1.0 else 0.0
        self.pan_x = max(-max_pan, min(max_pan, self.pan_x))
        self.pan_y = max(-max_pan, min(max_pan, self.pan_y))

    def reset_view(self):
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.hud_zoom_chip.configure(text="Zoom: 100%")
        self.redraw()

    def on_pan_start(self, event):
        self._pan_start = (event.x, event.y)
        self.canvas.configure(cursor="fleur")

    def on_pan_move(self, event):
        if self._pan_start is None or self.frame is None:
            return
        scale = self._view_params()[0]
        dx = (event.x - self._pan_start[0]) / scale
        dy = (event.y - self._pan_start[1]) / scale
        self._pan_start = (event.x, event.y)
        self.pan_x -= dx
        self.pan_y -= dy
        self._clamp_pan()
        self.redraw()

    def on_pan_end(self, _event):
        self._pan_start = None
        self.canvas.configure(cursor="crosshair")

    def toggle_loupe(self):
        self.show_loupe = not self.show_loupe
        text = "🔍 Loupe: ON" if self.show_loupe else "🔍 Loupe: OFF"
        color = self.theme["accent_emerald"] if self.show_loupe else self.theme["text_muted"]
        self.loupe_toggle_btn.configure(text=text, text_color=color)
        self.redraw()

    # -------------------------------------------------------------------------
    # ROI Editor Operations (New, Undo, Reset, Save, Delete)
    # -------------------------------------------------------------------------
    def update_roi_point_counters(self):
        count = len(self.points)
        self.roi_pts_counter_lbl.configure(text=f"{count}/{MAX_POINTS} Titik")
        if count == MAX_POINTS:
            valid, msg = polygon_is_valid(self.points)
            if valid:
                self.roi_pts_counter_lbl.configure(text_color=self.theme["accent_emerald"])
                self.set_status("✓ Poligon 4 titik lengkap. Periksa dimensi dan klik 'SIMPAN ROI'.", "ok")
            else:
                self.roi_pts_counter_lbl.configure(text_color=self.theme["accent_crimson"])
                self.set_status(f"⚠️ {msg}", "error")
        else:
            self.roi_pts_counter_lbl.configure(text_color=self.theme["accent_amber"])
            self.set_status(f"Klik titik ke-{count + 1} pada jalur video.", "info")
        self.update_telemetry_metrics()

    def update_telemetry_metrics(self):
        if len(self.points) == MAX_POINTS:
            px_area = polygon_area_px(self.points)
            try:
                l_m = float(self.roi_len_var.get().replace(",", "."))
                w_m = float(self.roi_wid_var.get().replace(",", "."))
                if l_m > 0 and w_m > 0:
                    real_area = l_m * w_m
                    ratio = l_m / w_m
                    self.telemetry_readout_lbl.configure(
                        text=f"Luas Tanah: {real_area:.1f} m²  |  Rasio P/L: {ratio:.2f}"
                    )
                    return
            except (ValueError, TypeError):
                pass
            self.telemetry_readout_lbl.configure(
                text=f"Luas Citra: {px_area:,.0f} px²  |  Dimensi belum lengkap"
            )
        else:
            self.telemetry_readout_lbl.configure(text="Luas Tanah: — m²  |  Rasio: —")

    def apply_dim_preset(self, length: float, width: float):
        self.roi_len_var.set(f"{length:g}")
        self.roi_wid_var.set(f"{width:g}")
        self.update_telemetry_metrics()
        self.redraw()

    def new_roi(self):
        if not self.entry:
            default_name = "roi_1"
        else:
            default_name = f"roi_{len(self.entry.get('rois', [])) + 1}"
        self.roi_name_var.set(default_name)
        self.roi_len_var.set("")
        self.roi_wid_var.set("")
        self.points = []
        self.active_roi_idx = None
        self.update_roi_point_counters()
        self.refresh_saved_rois_list()
        self.redraw()

    def undo_point(self):
        if self.points:
            self.points.pop()
            self.update_roi_point_counters()
            self.redraw()

    def reset_points(self):
        self.points = []
        self.update_roi_point_counters()
        self.redraw()

    def save_roi(self):
        if self.frame is None or not self.entry:
            self.set_status("Muat video CCTV terlebih dahulu.", "warn")
            return
        if len(self.points) != MAX_POINTS:
            self.set_status(f"Poligon belum lengkap ({len(self.points)}/{MAX_POINTS} titik).", "warn")
            return

        valid, reason = polygon_is_valid(self.points)
        if not valid:
            self.set_status(reason, "error")
            messagebox.showwarning("Poligon Tidak Valid", reason)
            return

        try:
            length_m = float(self.roi_len_var.get().replace(",", "."))
            width_m = float(self.roi_wid_var.get().replace(",", "."))
            if length_m <= 0 or width_m <= 0:
                raise ValueError
        except (ValueError, TypeError):
            self.set_status("Panjang dan Lebar harus berupa angka lebih dari 0 (meter).", "error")
            messagebox.showwarning("Dimensi Tidak Valid", "Masukkan angka positif untuk Panjang dan Lebar.")
            return

        name = self.roi_name_var.get().strip() or f"roi_{len(self.entry['rois']) + 1}"

        if self.active_roi_idx is not None and 0 <= self.active_roi_idx < len(self.entry["rois"]):
            self.entry["rois"][self.active_roi_idx] = {
                "roi_idx": self.active_roi_idx,
                "name": name,
                "polygon": [[int(x), int(y)] for x, y in self.points],
                "length_m": length_m,
                "width_m": width_m,
            }
            action_msg = f"ROI '{name}' berhasil diperbarui"
        else:
            new_idx = len(self.entry["rois"])
            self.entry["rois"].append({
                "roi_idx": new_idx,
                "name": name,
                "polygon": [[int(x), int(y)] for x, y in self.points],
                "length_m": length_m,
                "width_m": width_m,
            })
            action_msg = f"ROI '{name}' berhasil disimpan"

        save_json(self.store)
        self.refresh_video_list()
        self.refresh_saved_rois_list()
        self.update_header_stats()
        self.set_status(f"✓ {action_msg} ({length_m:g}m × {width_m:g}m). Tekan '+ Baru' untuk ROI berikutnya.", "ok")
        self.new_roi()

    def refresh_saved_rois_list(self):
        for child in self.saved_rois_list_frame.winfo_children():
            child.destroy()

        if not self.entry or not self.entry.get("rois"):
            self.saved_rois_count_lbl.configure(text="0 Item")
            no_lbl = tk.Label(
                self.saved_rois_list_frame, text="Belum ada ROI tersimpan di CCTV ini.",
                font=("Segoe UI", 9), bg=self.theme["card_bg"],
                fg=self.theme["text_muted"]
            )
            no_lbl.pack(pady=6)
            return

        rois = self.entry["rois"]
        self.saved_rois_count_lbl.configure(text=f"{len(rois)} Item")

        for r in rois:
            rid = r["roi_idx"]
            is_selected = (self.active_roi_idx == rid)

            card_bg = self.theme["card_bg_active"] if is_selected else self.theme["panel_bg"]
            card = tk.Frame(
                self.saved_rois_list_frame,
                bg=card_bg,
                highlightthickness=1,
                highlightbackground=self.theme["accent_emerald"] if is_selected else self.theme["card_border"],
                cursor="hand2"
            )
            card.pack(fill="x", pady=2)

            inner = tk.Frame(card, bg=card_bg)
            inner.pack(fill="both", expand=True, padx=8, pady=4)

            chip = tk.Label(
                inner, text="■", font=("Segoe UI", 10),
                bg=card_bg, fg=self.theme["accent_emerald"] if is_selected else self.theme["accent_blue"]
            )
            chip.pack(side="left", padx=(0, 6))

            txt = f"{r['name']} ({r['length_m']:g}×{r['width_m']:g}m)"
            lbl = tk.Label(
                inner, text=txt,
                font=("Segoe UI", 9, "bold" if is_selected else "normal"),
                bg=card_bg, fg=self.theme["text_primary"]
            )
            lbl.pack(side="left")

            del_btn = tk.Button(
                inner, text="✕", font=("Segoe UI", 8, "bold"),
                bg=card_bg, fg=self.theme["accent_crimson"],
                activebackground=self.theme["accent_crimson"],
                activeforeground="#FFFFFF",
                bd=0, relief="flat", padx=4, pady=0,
                command=lambda idx=rid: self.delete_roi(idx)
            )
            del_btn.pack(side="right")

            def on_sel(_e, idx=rid):
                self.select_saved_roi(idx)

            for w in (card, inner, chip, lbl):
                w.bind("<Button-1>", on_sel)

    def select_saved_roi(self, idx: int):
        if not self.entry or not self.entry.get("rois"):
            return
        roi = next((r for r in self.entry["rois"] if r["roi_idx"] == idx), None)
        if roi is None:
            return
        self.active_roi_idx = idx
        self.points = [list(p) for p in roi["polygon"]]
        self.roi_name_var.set(roi["name"])
        self.roi_len_var.set(f"{roi['length_m']:g}")
        self.roi_wid_var.set(f"{roi['width_m']:g}")
        self.update_roi_point_counters()
        self.refresh_saved_rois_list()
        self.set_status(f"Menampilkan ROI '{roi['name']}'. Anda dapat menggeser titiknya lalu klik Simpan.", "info")
        self.redraw()

    def delete_roi(self, idx: int):
        if not self.entry or not self.entry.get("rois"):
            return
        roi = next((r for r in self.entry["rois"] if r["roi_idx"] == idx), None)
        if roi is None:
            return

        ok = messagebox.askyesno(
            "Konfirmasi Hapus ROI",
            f"Apakah Anda yakin ingin menghapus ROI '{roi['name']}' dari {self.cctv_id}?\nTindakan ini tidak dapat dibatalkan."
        )
        if not ok:
            return

        self.entry["rois"].remove(roi)
        for i, r in enumerate(self.entry["rois"]):
            r["roi_idx"] = i

        save_json(self.store)
        self.refresh_video_list()
        self.refresh_saved_rois_list()
        self.update_header_stats()
        self.new_roi()
        self.set_status(f"✓ ROI '{roi['name']}' berhasil dihapus.", "ok")

    # -------------------------------------------------------------------------
    # Camera Perspective Slider & Presets
    # -------------------------------------------------------------------------
    def on_persp_slider_change(self, val):
        deg = float(val)
        self.set_perspective(deg)

    def set_perspective(self, deg: float):
        deg = round(deg, 1) % 360
        self.persp_slider.set(deg)
        self.persp_deg_val_lbl.configure(text=f"{deg:g}°", text_color=self.theme["accent_purple"])
        if self.entry is not None:
            self.entry["perspective_deg"] = deg
            save_json(self.store)
        self.redraw()

    # -------------------------------------------------------------------------
    # Status & Header Metrics
    # -------------------------------------------------------------------------
    def set_status(self, msg: str, kind: str = "info"):
        icons = {
            "info": "ℹ️",
            "ok": "✅",
            "warn": "⚠️",
            "error": "❌",
        }
        colors = {
            "info": self.theme["text_secondary"],
            "ok": self.theme["accent_emerald"],
            "warn": self.theme["accent_amber"],
            "error": self.theme["accent_crimson"],
        }
        self.status_icon_lbl.configure(text=icons.get(kind, "ℹ️"))
        self.status_text_lbl.configure(text=msg, text_color=colors.get(kind, self.theme["text_secondary"]))

    def update_header_stats(self):
        all_vids = list_videos()
        tot_vids = len(all_vids)
        annotated_count = sum(1 for e in self.store.values() if e.get("rois"))
        total_rois = sum(len(e.get("rois", [])) for e in self.store.values())

        self.stat_videos_chip.configure(text=f"📹 {tot_vids} Videos")
        self.stat_annotated_chip.configure(text=f"✓ {annotated_count} Dianotasi")
        self.stat_rois_chip.configure(text=f"🏷️ {total_rois} Total ROI")

    # -------------------------------------------------------------------------
    # Snapshot & Tools
    # -------------------------------------------------------------------------
    def export_snapshot(self, silent: bool = False):
        if self.frame is None:
            if not silent:
                messagebox.showinfo("Snapshot", "Muat video CCTV terlebih dahulu sebelum mengambil snapshot.")
            return None
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"snapshot_{self.cctv_id}_{ts}.png" if self.cctv_id else f"snapshot_{ts}.png"
        out_path = os.path.join(SNAPSHOT_DIR, name)

        rendered = self._render_annotated_frame(self.frame.copy())
        cv2.imwrite(out_path, rendered)
        self.set_status(f"✓ Snapshot disimpan ke: {os.path.basename(out_path)}", "ok")
        if not silent:
            messagebox.showinfo("Snapshot Berhasil", f"Snapshot frame berhasil diekspor ke:\n{out_path}")
        return out_path

    def toggle_theme(self):
        if self.theme_mode == "dark":
            self.theme_mode = "light"
            self.theme = THEME_LIGHT
            ctk.set_appearance_mode("light")
        else:
            self.theme_mode = "dark"
            self.theme = THEME_DARK
            ctk.set_appearance_mode("dark")
        self.configure(fg_color=self.theme["bg_main"])
        self.canvas.configure(bg=self.theme["canvas_bg"])
        self.refresh_video_list()
        self.refresh_saved_rois_list()
        self.redraw()

    def show_shortcuts_modal(self):
        shortcuts = (
            "📌 PINTASAN KEYBOARD ROI STUDIO\n\n"
            "• [Spasi]         : Putar / Jeda video\n"
            "• [Panah Kiri/Kanan] : Mundur / Maju 1 frame\n"
            "• [Shift + Kiri/Kanan] : Mundur / Maju 10 frame\n"
            "• [Klik Kiri]     : Letakkan titik / Drag titik sudut poligon\n"
            "• [Scroll Mouse]  : Zoom in / Zoom out ke posisi kursor\n"
            "• [Drag Kanan]    : Geser (pan) kanvas\n"
            "• [Ctrl + Z]      : Batalkan titik terakhir (Undo)\n"
            "• [Ctrl + S]      : Simpan ROI\n"
            "• [Ctrl + N]      : Buat ROI Baru\n"
            "• [Esc / Ctrl+R]  : Reset semua titik\n"
            "• [Ctrl + 0]      : Reset tampilan / Zoom fit\n"
        )
        messagebox.showinfo("Pintasan Keyboard & Panduan", shortcuts)

    # -------------------------------------------------------------------------
    # Core Viewport Rendering Pipeline
    # -------------------------------------------------------------------------
    def redraw(self):
        self.canvas.delete("all")
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)

        # Empty State
        if self.frame is None:
            cx, cy = cw // 2, ch // 2
            self.canvas.create_text(
                cx, cy - 35, text="🛰️ BELUM ADA CCTV DIMUAT",
                fill="#3B82F6", font=("Segoe UI", 16, "bold")
            )
            self.canvas.create_text(
                cx, cy + 5, text="Pilih video dari panel kiri lalu tekan 'Muat Video Terpilih'",
                fill="#94A3B8", font=("Segoe UI", 11)
            )
            self.canvas.create_text(
                cx, cy + 30, text="Klik 4 titik pada lajur jalan untuk membuat batas pengukuran kecepatan.",
                fill="#64748B", font=("Segoe UI", 10)
            )
            return

        h, w = self.frame.shape[:2]
        scale, ox, oy = self._view_params()
        disp_w, disp_h = int(w * scale), int(h * scale)
        ox_i, oy_i = int(round(ox)), int(round(oy))

        # Render display image at exact scaled resolution
        disp = cv2.resize(
            self.frame, (disp_w, disp_h),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        )

        # Helper: Convert native frame coordinate -> display image pixel coordinate (WITHOUT canvas ox/oy)
        def to_disp(p):
            return (int(round(p[0] * scale)), int(round(p[1] * scale)))

        pts_d = [to_disp(p) for p in self.points]

        # 1. Draw other saved ROIs
        if self.entry and self.entry.get("rois"):
            for saved_roi in self.entry["rois"]:
                if self.active_roi_idx == saved_roi["roi_idx"]:
                    continue
                s_pts = [to_disp(p) for p in saved_roi["polygon"]]
                cv2.polylines(disp, [np.array(s_pts)], True, (100, 100, 100), 2, cv2.LINE_AA)
                mx = int(np.mean([p[0] for p in s_pts]))
                my = int(np.mean([p[1] for p in s_pts]))
                cv2.putText(disp, saved_roi["name"], (mx - 15, my),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # 2. Draw Active Polygon Fill & Edges
        if len(pts_d) == MAX_POINTS:
            overlay = disp.copy()
            cv2.fillPoly(overlay, [np.array(pts_d)], CLR_FILL_BGR)
            disp = cv2.addWeighted(disp, 0.85, overlay, 0.15, 0)

        # Width edges: 1->2 & 3->4 (Amber Gold)
        width_val = self.roi_wid_var.get().strip()
        w_lbl = f"LEBAR {width_val}m" if width_val else "LEBAR"
        edges_width = [(0, 1), (2, 3)]
        for a, b in edges_width:
            if b < len(pts_d):
                cv2.line(disp, pts_d[a], pts_d[b], CLR_WIDTH_BGR, 3, cv2.LINE_AA)
                mx, my = (pts_d[a][0] + pts_d[b][0]) // 2, (pts_d[a][1] + pts_d[b][1]) // 2
                self._draw_label_pill(disp, w_lbl, mx, my - 12, CLR_WIDTH_BGR, (15, 23, 42))

        # Length edges: 2->3 & 4->1 (Electric Blue)
        len_val = self.roi_len_var.get().strip()
        l_lbl = f"PANJANG {len_val}m" if len_val else "PANJANG"
        if len(pts_d) >= 3:
            cv2.line(disp, pts_d[1], pts_d[2], CLR_LENGTH_BGR, 3, cv2.LINE_AA)
            mx, my = (pts_d[1][0] + pts_d[2][0]) // 2, (pts_d[1][1] + pts_d[2][1]) // 2
            self._draw_label_pill(disp, l_lbl, mx, my - 12, CLR_LENGTH_BGR, (15, 23, 42))
        if len(pts_d) == MAX_POINTS:
            cv2.line(disp, pts_d[3], pts_d[0], CLR_LENGTH_BGR, 3, cv2.LINE_AA)
            mx, my = (pts_d[3][0] + pts_d[0][0]) // 2, (pts_d[3][1] + pts_d[0][0]) // 2
            self._draw_label_pill(disp, l_lbl, mx, my - 12, CLR_LENGTH_BGR, (15, 23, 42))

        # 3. Perspective Vector Arrow
        if self.entry and self.entry.get("perspective_deg") is not None:
            persp = self.entry["perspective_deg"]
            cx_p, cy_p = disp_w // 2, disp_h // 2
            rad = np.deg2rad(persp)
            len_arrow = min(disp_w, disp_h) // 6
            tip = (int(cx_p + len_arrow * np.cos(rad)), int(cy_p - len_arrow * np.sin(rad)))
            cv2.arrowedLine(disp, (cx_p, cy_p), tip, CLR_PERSP_BGR, 3, cv2.LINE_AA, tipLength=0.25)
            self._draw_label_pill(disp, f"PERSPEKTIF {persp:g}°", tip[0] + 12, tip[1], CLR_PERSP_BGR, (15, 23, 42))

        # 4. Interactive Vertex Handles (Numbered & Glow Rings)
        for i, p in enumerate(pts_d):
            is_hovered = (self.hover_vertex_idx == i or self.drag_vertex_idx == i)
            radius = 12 if is_hovered else 9

            cv2.circle(disp, p, radius + 3, (15, 23, 42), -1, cv2.LINE_AA)
            cv2.circle(disp, p, radius, CLR_ACTIVE_BGR if is_hovered else CLR_VERTEX_BGR, -1, cv2.LINE_AA)
            cv2.circle(disp, p, radius, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(disp, str(i + 1), (p[0] - 4, p[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # 5. Magnifier / Loupe Inset (When cursor is over frame)
        if self.show_loupe and self.mouse_frame_pos is not None:
            fx, fy = self.mouse_frame_pos
            self._draw_loupe_inset(disp, fx, fy, cw, ch)

        img_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        self._photo_image = ImageTk.PhotoImage(pil_img, master=self.canvas)
        self.canvas.create_image(ox_i, oy_i, image=self._photo_image, anchor="nw")

    def _draw_label_pill(self, img, text: str, x: int, y: int, border_color_bgr, bg_color_bgr):
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        pad_x, pad_y = 6, 4
        x0, y0 = x - pad_x, y - th - pad_y
        x1, y1 = x + tw + pad_x, y + baseline + pad_y

        h, w = img.shape[:2]
        if not (0 <= x0 < w and 0 <= x1 < w and 0 <= y0 < h and 0 <= y1 < h):
            return

        cv2.rectangle(img, (x0, y0), (x1, y1), bg_color_bgr, -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), border_color_bgr, 1, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_loupe_inset(self, disp, fx: int, fy: int, cw: int, ch: int):
        loupe_size = 110
        loupe_zoom = 3
        crop_radius = loupe_size // (2 * loupe_zoom)

        h, w = self.frame.shape[:2]
        x0 = max(0, fx - crop_radius)
        y0 = max(0, fy - crop_radius)
        x1 = min(w, fx + crop_radius)
        y1 = min(h, fy + crop_radius)

        patch = self.frame[y0:y1, x0:x1]
        if patch.size == 0 or patch.shape[0] < 2 or patch.shape[1] < 2:
            return

        mag = cv2.resize(patch, (loupe_size, loupe_size), interpolation=cv2.INTER_NEAREST)

        mid = loupe_size // 2
        cv2.line(mag, (mid, 0), (mid, loupe_size), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(mag, (0, mid), (loupe_size, mid), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(mag, (mid, mid), 4, (0, 255, 0), 1, cv2.LINE_AA)

        lx, ly = max(0, disp.shape[1] - loupe_size - 12), 12
        if lx + loupe_size <= disp.shape[1] and ly + loupe_size <= disp.shape[0]:
            cv2.rectangle(disp, (lx - 2, ly - 2), (lx + loupe_size + 2, ly + loupe_size + 2), (15, 23, 42), -1)
            cv2.rectangle(disp, (lx - 2, ly - 2), (lx + loupe_size + 2, ly + loupe_size + 2), CLR_ACTIVE_BGR, 2, cv2.LINE_AA)
            disp[ly:ly + loupe_size, lx:lx + loupe_size] = mag
            cv2.putText(disp, "LOUPE 3X", (lx + 6, ly + loupe_size - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

    def _render_annotated_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        out = frame.copy()

        if self.entry and self.entry.get("rois"):
            for roi in self.entry["rois"]:
                pts = np.array(roi["polygon"], dtype=np.int32)
                overlay = out.copy()
                cv2.fillPoly(overlay, [pts], CLR_FILL_BGR)
                out = cv2.addWeighted(out, 0.85, overlay, 0.15, 0)
                cv2.polylines(out, [pts], True, CLR_LENGTH_BGR, 3, cv2.LINE_AA)

                mx = int(np.mean([p[0] for p in pts]))
                my = int(np.mean([p[1] for p in pts]))
                txt = f"{roi['name']} ({roi['length_m']}m × {roi['width_m']}m)"
                cv2.putText(out, txt, (mx - 20, my), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        badge_txt = f"BANDUNG CCTV ROI TELEMETRY • {self.video_name or ''}"
        cv2.putText(out, badge_txt, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, badge_txt, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (59, 130, 246), 2, cv2.LINE_AA)
        return out


# ==========================================
# Application Entry Point
# ==========================================
def main():
    videos = list_videos()
    if not videos:
        root = ctk.CTk()
        root.withdraw()
        messagebox.showerror(
            "Folder Video Kosong",
            f"Tidak ditemukan berkas video (.mp4, .avi, dll) pada direktori:\n{VIDEO_DIR}"
        )
        sys.exit(1)

    app = ModernRoiMakerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
