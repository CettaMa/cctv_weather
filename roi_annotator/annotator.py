#!/usr/bin/env python3
"""
CCTV ROI Annotator — Standalone Tool with Video Preview & Dimensioned 4-Point ROI.

Features:
- Loads video samples (.mp4/.avi/.mkv/.mov) or images (.jpg/.png) from ./samples/
- Built-in video player: Play/Pause (Space), scrub slider, and frame stepping (-10/-1/+1/+10)
- Input Panjang (m) and Lebar (m) for each ROI
- Color-coded edges on canvas & preview:
  - Biru (#38BDF8) = Lebar (Width)
  - Oranye (#F59E0B) = Panjang (Length)
  - Plus "Tukar Sisi Panjang <-> Lebar" button to flip edge roles anytime
- Left-click to place vertices; Right-click / Enter / "Selesai ROI" to close polygon
- Drag any vertex on a saved ROI to fine-tune its position
- Stores pixel coords, normalized coords, width_m, length_m, and orientation in ./outputs/roi_annotations.json
"""

import json
import os
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
ANNOTATIONS_FILE = os.path.join(OUTPUT_DIR, "roi_annotations.json")
PREVIEWS_DIR = os.path.join(OUTPUT_DIR, "previews")
METADATA_FILE = os.path.join(BASE_DIR, "package_info.json")

COLOR_WIDTH_HEX = "#38BDF8"   # Biru terang -> Lebar (Width)
COLOR_LENGTH_HEX = "#F59E0B"  # Oranye -> Panjang (Length)
COLOR_WIDTH_BGR = (248, 189, 56)
COLOR_LENGTH_BGR = (11, 158, 245)

ROI_COLORS = [
    "#69F0AE",
    "#E040FB",
    "#FF5252",
    "#448AFF",
    "#B2FF59",
]

VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
MEDIA_EXTS = VIDEO_EXTS + IMAGE_EXTS


class ROIAnnotatorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CCTV ROI Annotator")
        self.root.geometry("1420x900")
        self.root.minsize(1080, 720)
        self.root.configure(bg="#0F172A")

        os.makedirs(SAMPLES_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(PREVIEWS_DIR, exist_ok=True)

        self.package_info = self._load_package_info()
        if self.package_info.get("package_name"):
            self.root.title(f"CCTV ROI Annotator — {self.package_info['package_name']}")

        self.media_files = self._scan_media()
        self.annotations = self._load_annotations()

        self.current_idx = 0
        self.cap: cv2.VideoCapture | None = None
        self.is_video = False
        self.total_frames = 1
        self.current_frame_idx = 0
        self.video_fps = 25.0
        self.is_playing = False
        self._play_job: str | None = None
        self._updating_slider = False

        self.current_bgr: np.ndarray | None = None
        self.current_pil: Image.Image | None = None
        self.current_tk: ImageTk.PhotoImage | None = None
        self.img_w = 1
        self.img_h = 1
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.draft_points: list[tuple[int, int]] = []
        self.selected_roi_idx: int | None = None
        self.drag_target: tuple[int, int] | None = None

        self.four_point_mode = tk.BooleanVar(value=True)
        self.annotator_name = tk.StringVar(
            value=self.package_info.get("annotator", "Annotator")
        )
        self.roi_label_var = tk.StringVar(value="ROI_1")
        self.roi_width_var = tk.StringVar(value="3.5")
        self.roi_length_var = tk.StringVar(value="15.0")

        self._build_ui()
        self._bind_events()

        if self.media_files:
            self.load_media(0)
        else:
            messagebox.showwarning(
                "Tidak Ada Media",
                f"Folder samples kosong:\n{SAMPLES_DIR}\n"
                "Pastikan video/gambar sample CCTV sudah ada di folder samples/.",
            )

    def _load_package_info(self) -> dict:
        if os.path.exists(METADATA_FILE):
            try:
                with open(METADATA_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _scan_media(self) -> list[str]:
        return [
            f
            for f in sorted(os.listdir(SAMPLES_DIR))
            if f.lower().endswith(MEDIA_EXTS)
        ]

    def _load_annotations(self) -> dict:
        if os.path.exists(ANNOTATIONS_FILE):
            try:
                with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "cameras" in data:
                    return data
            except Exception:
                pass
        return {
            "package_name": self.package_info.get("package_name", "all"),
            "annotator": self.package_info.get("annotator", ""),
            "updated_at": "",
            "cameras": {},
        }

    def _save_annotations(self, show_toast: bool = False):
        self.annotations["annotator"] = self.annotator_name.get().strip()
        self.annotations["updated_at"] = datetime.now().isoformat(timespec="seconds")
        with open(ANNOTATIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.annotations, f, indent=2, ensure_ascii=False)
        if show_toast:
            self.status_var.set(f"Tersimpan ke {os.path.basename(ANNOTATIONS_FILE)}")

    @staticmethod
    def _parse_dim(val_str: str, default: float) -> float:
        try:
            v = float(val_str.strip().replace(",", "."))
            return round(v, 3) if v > 0 else default
        except ValueError:
            return default

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")

        main_pane = tk.PanedWindow(
            self.root, orient=tk.HORIZONTAL, bg="#0F172A", sashwidth=4, bd=0
        )
        main_pane.pack(fill=tk.BOTH, expand=True)

        # --- Left Sidebar: CCTV List ---
        left_frame = tk.Frame(main_pane, bg="#1E293B", width=260)
        main_pane.add(left_frame, minsize=220)

        tk.Label(
            left_frame,
            text="Daftar CCTV",
            bg="#1E293B",
            fg="#38BDF8",
            font=("Segoe UI", 12, "bold"),
            pady=10,
        ).pack(fill=tk.X, padx=10)

        self.progress_var = tk.StringVar(value="0 / 0 selesai")
        tk.Label(
            left_frame,
            textvariable=self.progress_var,
            bg="#1E293B",
            fg="#94A3B8",
            font=("Segoe UI", 9),
        ).pack(fill=tk.X, padx=10, pady=(0, 6))

        list_container = tk.Frame(left_frame, bg="#1E293B")
        list_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        scrollbar = tk.Scrollbar(list_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.cam_listbox = tk.Listbox(
            list_container,
            bg="#0F172A",
            fg="#E2E8F0",
            selectbackground="#2563EB",
            selectforeground="#FFFFFF",
            font=("Consolas", 9),
            activestyle="none",
            bd=0,
            highlightthickness=1,
            highlightbackground="#334155",
            yscrollcommand=scrollbar.set,
        )
        self.cam_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.cam_listbox.yview)

        nav_frame = tk.Frame(left_frame, bg="#1E293B")
        nav_frame.pack(fill=tk.X, padx=8, pady=(0, 10))
        tk.Button(
            nav_frame,
            text="<< Prev CCTV",
            command=self.prev_media,
            bg="#334155",
            fg="#F8FAFC",
            relief=tk.FLAT,
            pady=6,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        tk.Button(
            nav_frame,
            text="Next CCTV >>",
            command=self.next_media,
            bg="#2563EB",
            fg="#FFFFFF",
            relief=tk.FLAT,
            pady=6,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        # --- Center: Canvas + Video Controls ---
        center_frame = tk.Frame(main_pane, bg="#0F172A")
        main_pane.add(center_frame, minsize=560)

        top_bar = tk.Frame(center_frame, bg="#1E293B", height=42)
        top_bar.pack(fill=tk.X)

        self.title_var = tk.StringVar(value="Tidak ada media")
        tk.Label(
            top_bar,
            textvariable=self.title_var,
            bg="#1E293B",
            fg="#F8FAFC",
            font=("Segoe UI", 11, "bold"),
            padx=12,
            pady=8,
        ).pack(side=tk.LEFT)

        tk.Label(
            top_bar,
            text="Klik kiri: tambah titik | Drag titik: geser | Space: Play/Pause",
            bg="#1E293B",
            fg="#94A3B8",
            font=("Segoe UI", 9),
            padx=12,
        ).pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(
            center_frame, bg="#0B1120", highlightthickness=0, cursor="crosshair"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Video Player Bar
        player_bar = tk.Frame(center_frame, bg="#1E293B", pady=6, padx=10)
        player_bar.pack(fill=tk.X, padx=6, pady=(0, 4))

        self.play_btn = tk.Button(
            player_bar,
            text="Play",
            command=self.toggle_play,
            bg="#059669",
            fg="#FFFFFF",
            relief=tk.FLAT,
            width=7,
            font=("Segoe UI", 9, "bold"),
        )
        self.play_btn.pack(side=tk.LEFT, padx=(0, 6))

        for label, delta in [("-10", -10), ("-1", -1), ("+1", 1), ("+10", 10)]:
            tk.Button(
                player_bar,
                text=label,
                command=lambda d=delta: self.step_frame(d),
                bg="#334155",
                fg="#F8FAFC",
                relief=tk.FLAT,
                width=4,
                font=("Segoe UI", 8, "bold"),
            ).pack(side=tk.LEFT, padx=2)

        self.frame_slider = tk.Scale(
            player_bar,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            showvalue=False,
            bg="#1E293B",
            fg="#F8FAFC",
            troughcolor="#0F172A",
            activebackground="#38BDF8",
            highlightthickness=0,
            command=self._on_slider_move,
        )
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        self.frame_info_var = tk.StringVar(value="Frame 0 / 0")
        tk.Label(
            player_bar,
            textvariable=self.frame_info_var,
            bg="#1E293B",
            fg="#CBD5E1",
            font=("Consolas", 9),
            width=16,
            anchor="e",
        ).pack(side=tk.RIGHT)

        self.status_var = tk.StringVar(value="Siap")
        tk.Label(
            center_frame,
            textvariable=self.status_var,
            bg="#1E293B",
            fg="#CBD5E1",
            anchor="w",
            padx=10,
            pady=4,
            font=("Segoe UI", 9),
        ).pack(fill=tk.X)

        # --- Right Sidebar: Controls & ROI List ---
        right_frame = tk.Frame(main_pane, bg="#1E293B", width=310)
        main_pane.add(right_frame, minsize=280)

        tk.Label(
            right_frame,
            text="Pengaturan & Dimensi ROI",
            bg="#1E293B",
            fg="#38BDF8",
            font=("Segoe UI", 12, "bold"),
            pady=8,
        ).pack(fill=tk.X, padx=10)

        tk.Label(
            right_frame,
            text="Nama Annotator:",
            bg="#1E293B",
            fg="#CBD5E1",
            anchor="w",
            font=("Segoe UI", 9),
        ).pack(fill=tk.X, padx=10)
        tk.Entry(
            right_frame,
            textvariable=self.annotator_name,
            bg="#0F172A",
            fg="#F8FAFC",
            insertbackground="#F8FAFC",
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground="#334155",
        ).pack(fill=tk.X, padx=10, pady=(2, 8), ipady=4)

        tk.Label(
            right_frame,
            text="Label ROI:",
            bg="#1E293B",
            fg="#CBD5E1",
            anchor="w",
            font=("Segoe UI", 9),
        ).pack(fill=tk.X, padx=10)
        tk.Entry(
            right_frame,
            textvariable=self.roi_label_var,
            bg="#0F172A",
            fg="#F8FAFC",
            insertbackground="#F8FAFC",
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground="#334155",
        ).pack(fill=tk.X, padx=10, pady=(2, 8), ipady=4)

        # Panjang & Lebar ROI Inputs
        dim_row = tk.Frame(right_frame, bg="#1E293B")
        dim_row.pack(fill=tk.X, padx=10, pady=(0, 6))

        col_w = tk.Frame(dim_row, bg="#1E293B")
        col_w.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        tk.Label(
            col_w,
            text="Lebar (m) [Biru]:",
            bg="#1E293B",
            fg=COLOR_WIDTH_HEX,
            anchor="w",
            font=("Segoe UI", 9, "bold"),
        ).pack(fill=tk.X)
        tk.Entry(
            col_w,
            textvariable=self.roi_width_var,
            bg="#0F172A",
            fg="#F8FAFC",
            insertbackground="#F8FAFC",
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=COLOR_WIDTH_HEX,
        ).pack(fill=tk.X, pady=(2, 0), ipady=4)

        col_l = tk.Frame(dim_row, bg="#1E293B")
        col_l.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        tk.Label(
            col_l,
            text="Panjang (m) [Oranye]:",
            bg="#1E293B",
            fg=COLOR_LENGTH_HEX,
            anchor="w",
            font=("Segoe UI", 9, "bold"),
        ).pack(fill=tk.X)
        tk.Entry(
            col_l,
            textvariable=self.roi_length_var,
            bg="#0F172A",
            fg="#F8FAFC",
            insertbackground="#F8FAFC",
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=COLOR_LENGTH_HEX,
        ).pack(fill=tk.X, pady=(2, 0), ipady=4)

        # Color Legend Box
        legend_frame = tk.Frame(
            right_frame,
            bg="#0F172A",
            highlightthickness=1,
            highlightbackground="#334155",
            padx=8,
            pady=6,
        )
        legend_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        tk.Label(
            legend_frame,
            text="Legenda Warna Sisi ROI:",
            bg="#0F172A",
            fg="#E2E8F0",
            font=("Segoe UI", 8, "bold"),
            anchor="w",
        ).pack(fill=tk.X)
        tk.Label(
            legend_frame,
            text="━━  Biru = Lebar (Titik 1-2 & 3-4)",
            bg="#0F172A",
            fg=COLOR_WIDTH_HEX,
            font=("Segoe UI", 8, "bold"),
            anchor="w",
        ).pack(fill=tk.X, pady=(2, 0))
        tk.Label(
            legend_frame,
            text="━━  Oranye = Panjang (Titik 2-3 & 4-1)",
            bg="#0F172A",
            fg=COLOR_LENGTH_HEX,
            font=("Segoe UI", 8, "bold"),
            anchor="w",
        ).pack(fill=tk.X)

        tk.Checkbutton(
            right_frame,
            text="Mode 4 Titik (otomatis tutup di titik ke-4)",
            variable=self.four_point_mode,
            bg="#1E293B",
            fg="#E2E8F0",
            selectcolor="#0F172A",
            activebackground="#1E293B",
            activeforeground="#FFFFFF",
            font=("Segoe UI", 9),
        ).pack(fill=tk.X, padx=8, pady=(0, 6))

        btn_specs = [
            ("Selesai ROI (Enter)", self.finish_draft_roi, "#059669"),
            ("Update Dimensi / Label ROI Terpilih", self.update_selected_roi_meta, "#0284C7"),
            ("Tukar Sisi Panjang <-> Lebar", self.swap_selected_roi_orientation, "#D97706"),
            ("Undo Titik (Ctrl+Z)", self.undo_point, "#475569"),
            ("Batal Draft (Esc)", self.cancel_draft, "#64748B"),
            ("Hapus ROI Terpilih (Del)", self.delete_selected_roi, "#DC2626"),
            ("Hapus Semua ROI CCTV Ini", self.clear_current_rois, "#991B1B"),
            ("Simpan & Export Preview (Ctrl+S)", self.save_and_export, "#2563EB"),
        ]
        for label, cmd, color in btn_specs:
            tk.Button(
                right_frame,
                text=label,
                command=cmd,
                bg=color,
                fg="#FFFFFF",
                relief=tk.FLAT,
                font=("Segoe UI", 9, "bold"),
                pady=5,
            ).pack(fill=tk.X, padx=10, pady=2)

        tk.Label(
            right_frame,
            text="Daftar ROI pada CCTV Ini:",
            bg="#1E293B",
            fg="#CBD5E1",
            anchor="w",
            font=("Segoe UI", 9, "bold"),
        ).pack(fill=tk.X, padx=10, pady=(10, 4))

        self.roi_listbox = tk.Listbox(
            right_frame,
            bg="#0F172A",
            fg="#E2E8F0",
            selectbackground="#2563EB",
            font=("Consolas", 9),
            bd=0,
            highlightthickness=1,
            highlightbackground="#334155",
            height=8,
        )
        self.roi_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self._refresh_cam_listbox()

    def _bind_events(self):
        self.cam_listbox.bind("<<ListboxSelect>>", self._on_cam_select)
        self.roi_listbox.bind("<<ListboxSelect>>", self._on_roi_select)
        self.canvas.bind("<Configure>", lambda e: self._render_canvas())
        self.canvas.bind("<Button-1>", self._on_left_down)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_up)
        self.canvas.bind("<Button-3>", lambda e: self.finish_draft_roi())
        self.canvas.bind("<Double-Button-1>", lambda e: self.finish_draft_roi())

        self.root.bind("<space>", self._on_space)
        self.root.bind("<Return>", self._on_enter)
        self.root.bind("<Escape>", lambda e: self.cancel_draft())
        self.root.bind("<Control-z>", lambda e: self.undo_point())
        self.root.bind("<Control-s>", lambda e: self.save_and_export())
        self.root.bind("<Delete>", self._on_delete)
        self.root.bind("<Left>", lambda e: self.step_frame(-1))
        self.root.bind("<Right>", lambda e: self.step_frame(1))
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_space(self, event):
        if isinstance(event.widget, tk.Entry):
            return
        self.toggle_play()

    def _on_enter(self, event):
        if isinstance(event.widget, tk.Entry):
            self.update_selected_roi_meta()
            return
        self.finish_draft_roi()

    def _on_delete(self, event):
        if isinstance(event.widget, tk.Entry):
            return
        self.delete_selected_roi()

    def _on_close(self):
        self.stop_play()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.root.destroy()

    def _refresh_cam_listbox(self):
        self.cam_listbox.delete(0, tk.END)
        done_count = 0
        for fname in self.media_files:
            rois = self.annotations["cameras"].get(fname, {}).get("rois", [])
            mark = f"[{len(rois)}]" if rois else "[ ]"
            if rois:
                done_count += 1
            self.cam_listbox.insert(tk.END, f"{mark} {fname}")
        self.progress_var.set(f"{done_count} / {len(self.media_files)} CCTV teranotasi")
        if self.media_files and self.current_idx < len(self.media_files):
            self.cam_listbox.selection_clear(0, tk.END)
            self.cam_listbox.selection_set(self.current_idx)
            self.cam_listbox.see(self.current_idx)

    def _refresh_roi_listbox(self):
        self.roi_listbox.delete(0, tk.END)
        rois = self._get_current_rois()
        for i, roi in enumerate(rois):
            w_m = roi.get("width_m", "-")
            l_m = roi.get("length_m", "-")
            self.roi_listbox.insert(
                tk.END, f"{i + 1}. {roi.get('label', 'ROI')} ({w_m}m x {l_m}m)"
            )
        if self.selected_roi_idx is not None and 0 <= self.selected_roi_idx < len(rois):
            self.roi_listbox.selection_set(self.selected_roi_idx)

    def _current_filename(self) -> str:
        return self.media_files[self.current_idx]

    def _get_current_rois(self) -> list[dict]:
        if not self.media_files:
            return []
        fname = self._current_filename()
        cam_entry = self.annotations["cameras"].get(fname)
        return cam_entry.get("rois", []) if cam_entry else []

    def _ensure_current_cam_entry(self):
        fname = self._current_filename()
        if fname not in self.annotations["cameras"]:
            self.annotations["cameras"][fname] = {
                "filename": fname,
                "width": self.img_w,
                "height": self.img_h,
                "reference_frame": self.current_frame_idx,
                "rois": [],
            }
        else:
            self.annotations["cameras"][fname]["width"] = self.img_w
            self.annotations["cameras"][fname]["height"] = self.img_h
            self.annotations["cameras"][fname]["reference_frame"] = self.current_frame_idx

    def load_media(self, idx: int):
        if not self.media_files:
            return
        self.stop_play()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        idx = max(0, min(idx, len(self.media_files) - 1))
        self.current_idx = idx
        self.draft_points.clear()
        self.selected_roi_idx = None
        self.drag_target = None

        fname = self._current_filename()
        path = os.path.join(SAMPLES_DIR, fname)
        self.is_video = fname.lower().endswith(VIDEO_EXTS)

        if self.is_video:
            self.cap = cv2.VideoCapture(path)
            total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.video_fps = fps if fps and 1 <= fps <= 60 else 25.0
            self.total_frames = max(1, total)
            saved_ref = self.annotations["cameras"].get(fname, {}).get("reference_frame")
            start_f = (
                int(saved_ref)
                if isinstance(saved_ref, int) and 0 <= saved_ref < self.total_frames
                else self.total_frames // 2
            )
            self._updating_slider = True
            self.frame_slider.configure(to=max(0, self.total_frames - 1))
            self._updating_slider = False
            self._seek_frame(start_f)
        else:
            self.total_frames = 1
            self.current_frame_idx = 0
            bgr = cv2.imread(path)
            self._set_frame_bgr(bgr)
            self._updating_slider = True
            self.frame_slider.configure(to=0)
            self.frame_slider.set(0)
            self._updating_slider = False
            self.frame_info_var.set("Image (1/1)")

        self.title_var.set(
            f"({idx + 1}/{len(self.media_files)}) {fname}  [{self.img_w}x{self.img_h}]"
        )
        next_num = len(self._get_current_rois()) + 1
        self.roi_label_var.set(f"ROI_{next_num}")
        self._refresh_cam_listbox()
        self._refresh_roi_listbox()
        self._render_canvas()
        self.status_var.set(f"Memuat {fname}")

    def _set_frame_bgr(self, bgr: np.ndarray | None):
        if bgr is None or bgr.size == 0:
            return
        self.current_bgr = bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.current_pil = Image.fromarray(rgb)
        self.img_w, self.img_h = self.current_pil.size

    def _seek_frame(self, frame_idx: int):
        if not self.is_video or self.cap is None:
            return
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, bgr = self.cap.read()
        if ret and bgr is not None:
            self.current_frame_idx = frame_idx
            self._set_frame_bgr(bgr)
            self._updating_slider = True
            self.frame_slider.set(frame_idx)
            self._updating_slider = False
            self.frame_info_var.set(f"Frame {frame_idx + 1} / {self.total_frames}")
            self._render_canvas()

    def _on_slider_move(self, val_str: str):
        if self._updating_slider or not self.is_video:
            return
        self.stop_play()
        self._seek_frame(int(float(val_str)))

    def step_frame(self, delta: int):
        if not self.is_video:
            return
        self.stop_play()
        self._seek_frame(self.current_frame_idx + delta)

    def toggle_play(self):
        if not self.is_video:
            return
        if self.is_playing:
            self.stop_play()
        else:
            self.is_playing = True
            self.play_btn.configure(text="Pause", bg="#D97706")
            self._play_loop()

    def stop_play(self):
        self.is_playing = False
        if self._play_job is not None:
            self.root.after_cancel(self._play_job)
            self._play_job = None
        if hasattr(self, "play_btn"):
            self.play_btn.configure(text="Play", bg="#059669")

    def _play_loop(self):
        if not self.is_playing or not self.is_video or self.cap is None:
            return
        nxt = self.current_frame_idx + 1
        if nxt >= self.total_frames:
            nxt = 0
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, bgr = self.cap.read()
        if ret and bgr is not None:
            self.current_frame_idx = nxt
            self._set_frame_bgr(bgr)
            self._updating_slider = True
            self.frame_slider.set(nxt)
            self._updating_slider = False
            self.frame_info_var.set(f"Frame {nxt + 1} / {self.total_frames}")
            self._render_canvas()
        delay_ms = max(15, int(1000 / self.video_fps))
        self._play_job = self.root.after(delay_ms, self._play_loop)

    def _canvas_to_image(self, cx: float, cy: float) -> tuple[int, int] | None:
        if self.current_pil is None or self.scale <= 0:
            return None
        ix = int(round((cx - self.offset_x) / self.scale))
        iy = int(round((cy - self.offset_y) / self.scale))
        return (
            max(0, min(ix, self.img_w - 1)),
            max(0, min(iy, self.img_h - 1)),
        )

    def _image_to_canvas(self, ix: int, iy: int) -> tuple[float, float]:
        return (self.offset_x + ix * self.scale, self.offset_y + iy * self.scale)

    @staticmethod
    def _edge_role(edge_idx: int, swapped: bool) -> str:
        # edge 0 (P1-P2) & edge 2 (P3-P4) are Width by default; edge 1 & 3 are Length
        is_width = (edge_idx % 2 == 0) ^ swapped
        return "width" if is_width else "length"

    def _render_canvas(self):
        self.canvas.delete("all")
        if self.current_pil is None:
            return
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)

        self.scale = min((cw - 16) / self.img_w, (ch - 16) / self.img_h)
        disp_w = max(1, int(self.img_w * self.scale))
        disp_h = max(1, int(self.img_h * self.scale))
        self.offset_x = (cw - disp_w) // 2
        self.offset_y = (ch - disp_h) // 2

        resample = (
            Image.Resampling.BILINEAR
            if self.is_playing
            else Image.Resampling.LANCZOS
        )
        resized = self.current_pil.resize((disp_w, disp_h), resample)
        self.current_tk = ImageTk.PhotoImage(resized)
        self.canvas.create_image(
            self.offset_x, self.offset_y, anchor=tk.NW, image=self.current_tk
        )

        # Draw saved ROIs
        rois = self._get_current_rois()
        for r_idx, roi in enumerate(rois):
            pts = roi.get("points_px", [])
            if len(pts) < 2:
                continue
            c_pts = [self._image_to_canvas(px, py) for px, py in pts]
            line_w = 4 if r_idx == self.selected_roi_idx else 3
            w_m = roi.get("width_m", 0)
            l_m = roi.get("length_m", 0)
            swapped = bool(roi.get("orientation_swapped", False))

            if len(c_pts) == 4:
                for e_idx in range(4):
                    p1 = c_pts[e_idx]
                    p2 = c_pts[(e_idx + 1) % 4]
                    role = self._edge_role(e_idx, swapped)
                    color = COLOR_WIDTH_HEX if role == "width" else COLOR_LENGTH_HEX
                    self.canvas.create_line(*p1, *p2, fill=color, width=line_w)
                    if e_idx in (0, 1):
                        mx = (p1[0] + p2[0]) / 2
                        my = (p1[1] + p2[1]) / 2
                        dim_txt = (
                            f"Lebar: {w_m}m" if role == "width" else f"Panjang: {l_m}m"
                        )
                        self.canvas.create_text(
                            mx,
                            my - 10,
                            text=dim_txt,
                            fill=color,
                            font=("Segoe UI", 9, "bold"),
                        )
            else:
                color = ROI_COLORS[r_idx % len(ROI_COLORS)]
                flat = [coord for pair in c_pts for coord in pair]
                self.canvas.create_polygon(*flat, outline=color, fill="", width=line_w)

            for p_idx, (px, py) in enumerate(c_pts):
                r = 6 if r_idx == self.selected_roi_idx else 4
                self.canvas.create_oval(
                    px - r, py - r, px + r, py + r, fill="#FFFFFF", outline="#0F172A"
                )
                self.canvas.create_text(
                    px + 10,
                    py - 10,
                    text=str(p_idx + 1),
                    fill="#F8FAFC",
                    font=("Segoe UI", 8, "bold"),
                )

            cx_avg = sum(p[0] for p in c_pts) / len(c_pts)
            cy_avg = sum(p[1] for p in c_pts) / len(c_pts)
            self.canvas.create_text(
                cx_avg,
                cy_avg,
                text=f"{roi.get('label', f'ROI_{r_idx + 1}')}\n({w_m}m x {l_m}m)",
                fill="#FFFFFF",
                font=("Segoe UI", 10, "bold"),
                justify=tk.CENTER,
            )

        # Draw draft ROI with live color-coded edges
        if self.draft_points:
            c_draft = [self._image_to_canvas(px, py) for px, py in self.draft_points]
            for i in range(len(c_draft) - 1):
                role = self._edge_role(i, False)
                color = COLOR_WIDTH_HEX if role == "width" else COLOR_LENGTH_HEX
                self.canvas.create_line(
                    *c_draft[i], *c_draft[i + 1], fill=color, width=3, dash=(6, 3)
                )
            for idx, (px, py) in enumerate(c_draft):
                self.canvas.create_oval(
                    px - 5, py - 5, px + 5, py + 5, fill="#FACC15", outline="#0F172A"
                )
                self.canvas.create_text(
                    px + 10,
                    py - 10,
                    text=str(idx + 1),
                    fill="#FACC15",
                    font=("Segoe UI", 9, "bold"),
                )

    def _find_vertex_hit(self, cx: float, cy: float) -> tuple[int, int] | None:
        rois = self._get_current_rois()
        for r_idx, roi in enumerate(rois):
            for p_idx, (ix, iy) in enumerate(roi.get("points_px", [])):
                vx, vy = self._image_to_canvas(ix, iy)
                if (vx - cx) ** 2 + (vy - cy) ** 2 <= 100:
                    return (r_idx, p_idx)
        return None

    def _on_left_down(self, event):
        if self.current_pil is None:
            return
        self.stop_play()
        if not self.draft_points:
            hit = self._find_vertex_hit(event.x, event.y)
            if hit is not None:
                self.drag_target = hit
                self.selected_roi_idx = hit[0]
                self._sync_inputs_from_selected()
                self._refresh_roi_listbox()
                self._render_canvas()
                return

        pt = self._canvas_to_image(event.x, event.y)
        if pt is None:
            return
        self.draft_points.append(pt)
        self.status_var.set(
            f"Titik {len(self.draft_points)} ditambahkan: ({pt[0]}, {pt[1]})"
        )
        if self.four_point_mode.get() and len(self.draft_points) == 4:
            self.finish_draft_roi()
        else:
            self._render_canvas()

    def _on_left_drag(self, event):
        if self.drag_target is None:
            return
        pt = self._canvas_to_image(event.x, event.y)
        if pt is None:
            return
        r_idx, p_idx = self.drag_target
        rois = self._get_current_rois()
        if 0 <= r_idx < len(rois) and 0 <= p_idx < len(rois[r_idx]["points_px"]):
            rois[r_idx]["points_px"][p_idx] = [pt[0], pt[1]]
            rois[r_idx]["points_norm"][p_idx] = [
                round(pt[0] / self.img_w, 6),
                round(pt[1] / self.img_h, 6),
            ]
            self._render_canvas()

    def _on_left_up(self, _event):
        if self.drag_target is not None:
            self.drag_target = None
            self._save_annotations(show_toast=True)

    def _sync_inputs_from_selected(self):
        rois = self._get_current_rois()
        if self.selected_roi_idx is not None and 0 <= self.selected_roi_idx < len(rois):
            roi = rois[self.selected_roi_idx]
            self.roi_label_var.set(roi.get("label", "ROI"))
            self.roi_width_var.set(str(roi.get("width_m", 3.5)))
            self.roi_length_var.set(str(roi.get("length_m", 15.0)))

    def finish_draft_roi(self):
        if len(self.draft_points) < 3:
            if self.draft_points:
                self.status_var.set("Minimal 3 titik untuk membuat poligon ROI.")
            return
        self._ensure_current_cam_entry()
        label = (
            self.roi_label_var.get().strip()
            or f"ROI_{len(self._get_current_rois()) + 1}"
        )
        width_m = self._parse_dim(self.roi_width_var.get(), 3.5)
        length_m = self._parse_dim(self.roi_length_var.get(), 15.0)

        pts_px = [[int(x), int(y)] for x, y in self.draft_points]
        pts_norm = [
            [round(x / self.img_w, 6), round(y / self.img_h, 6)]
            for x, y in self.draft_points
        ]
        roi_obj = {
            "label": label,
            "width_m": width_m,
            "length_m": length_m,
            "orientation_swapped": False,
            "points_px": pts_px,
            "points_norm": pts_norm,
        }
        rois = self._get_current_rois()
        rois.append(roi_obj)
        self.selected_roi_idx = len(rois) - 1
        self.draft_points.clear()
        self.roi_label_var.set(f"ROI_{len(rois) + 1}")
        self._save_annotations(show_toast=True)
        self._refresh_cam_listbox()
        self._refresh_roi_listbox()
        self._render_canvas()

    def update_selected_roi_meta(self):
        rois = self._get_current_rois()
        if self.selected_roi_idx is None or not (0 <= self.selected_roi_idx < len(rois)):
            self.status_var.set("Pilih ROI dari daftar terlebih dahulu untuk diupdate.")
            return
        roi = rois[self.selected_roi_idx]
        roi["label"] = self.roi_label_var.get().strip() or roi.get("label", "ROI")
        roi["width_m"] = self._parse_dim(self.roi_width_var.get(), roi.get("width_m", 3.5))
        roi["length_m"] = self._parse_dim(self.roi_length_var.get(), roi.get("length_m", 15.0))
        self._save_annotations(show_toast=True)
        self._refresh_roi_listbox()
        self._render_canvas()

    def swap_selected_roi_orientation(self):
        rois = self._get_current_rois()
        if self.selected_roi_idx is None or not (0 <= self.selected_roi_idx < len(rois)):
            self.status_var.set("Pilih ROI terlebih dahulu untuk menukar sisi Panjang/Lebar.")
            return
        roi = rois[self.selected_roi_idx]
        roi["orientation_swapped"] = not bool(roi.get("orientation_swapped", False))
        self._save_annotations(show_toast=True)
        self._render_canvas()
        self.status_var.set("Sisi Panjang <-> Lebar berhasil ditukar.")

    def undo_point(self):
        if self.draft_points:
            self.draft_points.pop()
            self._render_canvas()
            self.status_var.set("Undo titik terakhir.")

    def cancel_draft(self):
        if self.draft_points:
            self.draft_points.clear()
            self._render_canvas()
            self.status_var.set("Draft ROI dibatalkan.")

    def delete_selected_roi(self):
        rois = self._get_current_rois()
        if self.selected_roi_idx is None or not (0 <= self.selected_roi_idx < len(rois)):
            return
        removed = rois.pop(self.selected_roi_idx)
        self.selected_roi_idx = None
        self._save_annotations(show_toast=True)
        self._refresh_cam_listbox()
        self._refresh_roi_listbox()
        self._render_canvas()
        self.status_var.set(f"ROI '{removed.get('label')}' dihapus.")

    def clear_current_rois(self):
        rois = self._get_current_rois()
        if not rois:
            return
        if messagebox.askyesno("Konfirmasi", "Hapus semua ROI pada CCTV ini?"):
            rois.clear()
            self.selected_roi_idx = None
            self._save_annotations(show_toast=True)
            self._refresh_cam_listbox()
            self._refresh_roi_listbox()
            self._render_canvas()

    def save_and_export(self):
        self._save_annotations(show_toast=True)
        if not self.media_files or self.current_bgr is None:
            return
        fname = self._current_filename()
        stem = os.path.splitext(fname)[0]
        rois = self._get_current_rois()
        img = self.current_bgr.copy()
        overlay = img.copy()

        for r_idx, roi in enumerate(rois):
            pts = np.array(roi["points_px"], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (120, 180, 100))
            w_m = roi.get("width_m", 0)
            l_m = roi.get("length_m", 0)
            swapped = bool(roi.get("orientation_swapped", False))

            if len(pts) == 4:
                for e_idx in range(4):
                    p1 = tuple(int(v) for v in pts[e_idx])
                    p2 = tuple(int(v) for v in pts[(e_idx + 1) % 4])
                    role = self._edge_role(e_idx, swapped)
                    bgr = COLOR_WIDTH_BGR if role == "width" else COLOR_LENGTH_BGR
                    cv2.line(img, p1, p2, bgr, 3)
                    if e_idx in (0, 1):
                        mx = (p1[0] + p2[0]) // 2
                        my = (p1[1] + p2[1]) // 2
                        txt = f"Lebar: {w_m}m" if role == "width" else f"Panjang: {l_m}m"
                        cv2.putText(
                            img, txt, (mx - 35, my - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, bgr, 2,
                        )
            else:
                cv2.polylines(img, [pts], True, COLOR_WIDTH_BGR, 3)

            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            cv2.putText(
                img,
                f"{roi.get('label', 'ROI')} ({w_m}x{l_m}m)",
                (cx - 55, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        blended = cv2.addWeighted(overlay, 0.2, img, 0.8, 0)
        out_preview = os.path.join(PREVIEWS_DIR, f"preview_{stem}.jpg")
        cv2.imwrite(out_preview, blended)
        self.status_var.set(
            f"Tersimpan JSON & preview frame {self.current_frame_idx + 1} untuk {fname}"
        )

    def _on_cam_select(self, _event):
        sel = self.cam_listbox.curselection()
        if sel and sel[0] != self.current_idx:
            self.load_media(sel[0])

    def _on_roi_select(self, _event):
        sel = self.roi_listbox.curselection()
        if sel:
            self.selected_roi_idx = sel[0]
            self._sync_inputs_from_selected()
            self._render_canvas()

    def prev_media(self):
        if self.current_idx > 0:
            self.load_media(self.current_idx - 1)

    def next_media(self):
        if self.current_idx < len(self.media_files) - 1:
            self.load_media(self.current_idx + 1)


def main():
    root = tk.Tk()
    ROIAnnotatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
